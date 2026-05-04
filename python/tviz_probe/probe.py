from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable


def main() -> None:
    args = parse_args()
    try:
        payload = probe(args.model_file, args.factory, args.input_spec, args.trace)
    except Exception as error:
        print(f"tviz probe error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit tviz ModelIR for a PyTorch model factory.")
    parser.add_argument("model_file", help="Python file containing a model factory")
    parser.add_argument("--factory", help="Factory function that returns nn.Module")
    parser.add_argument("--input", dest="input_spec", help='Input spec like "float32[1,3,64,64]"')
    parser.add_argument(
        "--trace",
        choices=["structure", "hook"],
        default="hook",
        help="structure avoids a forward pass; hook captures execution order and shapes",
    )
    return parser.parse_args()


def probe(model_file: str, factory_name: str | None, input_spec: str | None, trace: str) -> dict[str, Any]:
    torch = import_torch()
    module = load_module(Path(model_file))
    factory_name, factory = resolve_factory(torch, module, factory_name)
    model = factory()
    model.eval()

    total_params, trainable_params = count_parameters(model)
    warnings: list[str] = []

    if trace == "hook":
        try:
            inputs = build_inputs(torch, module, factory_name, input_spec)
            nodes, edges = hook_trace(torch, model, inputs)
            input_infos = tensor_infos("input", inputs)
        except Exception as error:
            warnings.append(f"hook trace unavailable; fell back to structure mode: {error}")
            nodes, edges = structure_trace(model)
            input_infos = []
    else:
        nodes, edges = structure_trace(model)
        input_infos = []

    groups = discover_groups(model)
    if not nodes:
        warnings.append("no modules were discovered")

    return {
        "schema_version": "0.1",
        "model": {
            "name": model.__class__.__name__,
            "source": model_file,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "inputs": input_infos,
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "warnings": warnings,
    }


def import_torch() -> ModuleType:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is not installed. Install it with `python -m pip install torch`.") from error
    return torch


def load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise RuntimeError(f"model file does not exist: {path}")
    module_name = f"tviz_user_model_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def resolve_factory(
    torch: ModuleType,
    module: ModuleType,
    factory_name: str | None,
) -> tuple[str, Callable[[], Any]]:
    if factory_name:
        factory = getattr(module, factory_name, None)
        if factory is None or not callable(factory):
            raise RuntimeError(f"factory {factory_name!r} was not found or is not callable")
        return factory_name, factory

    registry = getattr(module, "MODEL_REGISTRY", None)
    default_model = getattr(module, "DEFAULT_MODEL", None)
    if isinstance(registry, dict) and isinstance(default_model, str):
        factory = registry.get(default_model)
        if callable(factory):
            return default_model, factory

    for candidate in ["build_model", "create_model", "get_model", "model", "make_model"]:
        factory = getattr(module, candidate, None)
        if callable(factory) and is_zero_arg_callable(factory):
            return candidate, factory

    if isinstance(registry, dict) and len(registry) == 1:
        name, factory = next(iter(registry.items()))
        if callable(factory):
            return str(name), factory

    if isinstance(registry, dict) and "default" in registry and callable(registry["default"]):
        return "default", registry["default"]

    module_cls = getattr(torch.nn, "Module")
    candidates: list[tuple[str, Callable[[], Any]]] = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(value, type) and issubclass(value, module_cls) and value is not module_cls:
            try:
                value()
            except TypeError:
                continue
            candidates.append((name, value))

    if len(candidates) == 1:
        return candidates[0]

    public_factories = [
        (name, value)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and callable(value)
        and name.endswith("_model")
        and is_zero_arg_callable(value)
    ]
    if len(public_factories) == 1:
        return public_factories[0]

    raise RuntimeError(
        "could not infer which model to load. Add a zero-argument build_model(), "
        "create_model(), get_model(), or pass --factory NAME."
    )


def is_zero_arg_callable(value: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        return False
    return True


def build_inputs(
    torch: ModuleType,
    user_module: ModuleType,
    factory_name: str,
    input_spec: str | None,
) -> tuple[Any, ...]:
    if input_spec:
        return (tensor_from_spec(torch, input_spec),)
    sample_input_for = getattr(user_module, "sample_input_for", None)
    if sample_input_for is not None:
        inputs = sample_input_for(factory_name)
        return tuple(inputs) if isinstance(inputs, tuple) else (inputs,)
    raise RuntimeError("hook trace needs --input unless the model file exposes sample_input_for(factory)")


def tensor_from_spec(torch: ModuleType, spec: str) -> Any:
    match = re.fullmatch(r"([A-Za-z0-9_]+)\[([0-9,\s]+)\]", spec.strip())
    if not match:
        raise RuntimeError(f"invalid input spec {spec!r}; expected dtype[dim,dim,...]")
    dtype_name, dims_raw = match.groups()
    shape = tuple(int(part.strip()) for part in dims_raw.split(",") if part.strip())
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RuntimeError(f"unknown torch dtype {dtype_name!r}")
    if dtype_name.startswith("int") or dtype_name in {"long", "int64"}:
        return torch.randint(0, 128, shape, dtype=dtype)
    if dtype_name == "bool":
        return torch.zeros(shape, dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def hook_trace(torch: ModuleType, model: Any, inputs: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    modules = dict(model.named_modules())
    leaf_paths = {
        path
        for path, module in modules.items()
        if path and not any(child_path.startswith(f"{path}.") for child_path in modules if child_path != path)
    }
    records: list[dict[str, Any]] = []
    handles = []

    def make_hook(path: str, module: Any):
        def hook(_module: Any, module_inputs: tuple[Any, ...], output: Any) -> None:
            started = perf_counter()
            record = module_record(path, module)
            record["input_shapes"] = shape_labels(module_inputs)
            record["output_shapes"] = shape_labels(output)
            record["elapsed_us"] = int((perf_counter() - started) * 1_000_000)
            records.append(record)

        return hook

    for path, module in modules.items():
        if path in leaf_paths:
            handles.append(module.register_forward_hook(make_hook(path, module)))

    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()

    nodes = []
    for index, record in enumerate(records):
        record["id"] = record["module_path"]
        record["depth"] = record["module_path"].count(".")
        record["repeated"] = 1
        record["attributes"]["order"] = str(index + 1)
        nodes.append(record)

    edges = [
        {"from": nodes[index]["id"], "to": nodes[index + 1]["id"], "kind": "data"}
        for index in range(len(nodes) - 1)
    ]
    add_detected_skip_edges(nodes, edges)
    return nodes, edges


def structure_trace(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes = []
    for path, module in model.named_modules():
        if not path:
            continue
        record = module_record(path, module)
        record["id"] = path
        record["depth"] = path.count(".")
        record["repeated"] = 1
        nodes.append(record)
    edges = [
        {"from": nodes[index]["id"], "to": nodes[index + 1]["id"], "kind": "structure"}
        for index in range(len(nodes) - 1)
    ]
    return nodes, edges


def module_record(path: str, module: Any) -> dict[str, Any]:
    params, trainable = count_parameters(module, recurse=False)
    kind = module.__class__.__name__
    return {
        "id": path,
        "label": label_for(path, kind),
        "kind": kind,
        "module_path": path,
        "params": params,
        "trainable_params": trainable,
        "input_shapes": [],
        "output_shapes": [],
        "attributes": module_attributes(module),
        "style": style_for(kind),
        "repeated": 1,
        "depth": path.count("."),
    }


def module_attributes(module: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for name in [
        "in_features",
        "out_features",
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "num_heads",
        "head_dim",
        "normalized_shape",
        "embedding_dim",
        "num_embeddings",
    ]:
        if hasattr(module, name):
            attrs[name] = compact_value(getattr(module, name))
    return attrs


def discover_groups(model: Any) -> list[dict[str, Any]]:
    groups = []
    for path, module in model.named_modules():
        if not path:
            continue
        children = list(module.named_children())
        if len(children) < 2:
            continue
        child_paths = [f"{path}.{name}" for name, _child in children]
        child_types = [child.__class__.__name__ for _name, child in children]
        repeated = longest_same_type_run(child_types)
        label = module.__class__.__name__
        if repeated > 1 and len(set(child_types)) == 1:
            label = child_types[0]
        groups.append(
            {
                "id": path,
                "label": label,
                "children": child_paths,
                "repeated": repeated,
            }
        )
    return groups


def longest_same_type_run(types: list[str]) -> int:
    longest = 1
    current = 1
    for previous, item in zip(types, types[1:]):
        if item == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def add_detected_skip_edges(nodes: list[dict[str, Any]], edges: list[dict[str, str]]) -> None:
    for index, node in enumerate(nodes):
        lowered = f"{node['module_path']} {node['kind']}".lower()
        if "residual" in lowered and index + 2 < len(nodes):
            edges.append({"from": nodes[index]["id"], "to": nodes[index + 2]["id"], "kind": "skip"})


def tensor_infos(prefix: str, values: tuple[Any, ...]) -> list[dict[str, Any]]:
    infos = []
    for index, value in enumerate(values):
        dtype = str(getattr(value, "dtype", "unknown")).replace("torch.", "")
        shape = [str(item) for item in getattr(value, "shape", [])]
        infos.append({"name": f"{prefix}{index}", "dtype": dtype, "shape": shape})
    return infos


def shape_labels(value: Any) -> list[str]:
    if hasattr(value, "shape"):
        return [shape_label(value)]
    if isinstance(value, (list, tuple)):
        labels = []
        for item in value:
            labels.extend(shape_labels(item))
        return labels
    if isinstance(value, dict):
        labels = []
        for item in value.values():
            labels.extend(shape_labels(item))
        return labels
    return [type(value).__name__]


def shape_label(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
    if shape is None:
        return type(value).__name__
    dims = "x".join(str(item) for item in shape)
    return f"{dims}:{dtype}" if dtype else dims


def label_for(path: str, kind: str) -> str:
    tail = path.rsplit(".", 1)[-1]
    if tail.isdigit():
        return kind
    if kind.lower() == tail.lower():
        return kind
    return f"{tail} · {kind}"


def style_for(kind: str) -> str:
    lowered = kind.lower()
    if "conv" in lowered:
        return "conv"
    if "batchnorm" in lowered or "layernorm" in lowered or "groupnorm" in lowered or "norm" in lowered:
        return "norm"
    if lowered in {"relu", "gelu", "silu", "tanh", "sigmoid"} or "activation" in lowered:
        return "activation"
    if "attention" in lowered:
        return "attention"
    if "linear" in lowered or "feedforward" in lowered or "mlp" in lowered:
        return "mlp"
    if "embedding" in lowered:
        return "embedding"
    if "pool" in lowered:
        return "pooling"
    if "flatten" in lowered or "head" in lowered or "classifier" in lowered:
        return "output"
    return "custom"


def count_parameters(module: Any, recurse: bool = True) -> tuple[int, int]:
    parameters = list(module.parameters(recurse=recurse))
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return int(total), int(trainable)


def compact_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
