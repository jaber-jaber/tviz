from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def main() -> None:
    args = parse_args()
    try:
        config, source = fetch_config(args.model, args.revision)
        payload = build_ir(args.model, source, config)
    except Exception as error:
        print(f"tviz hf error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit tviz ModelIR from a Hugging Face config.json.")
    parser.add_argument("model", help="Hugging Face repo id or URL")
    parser.add_argument("--revision", default="main", help="branch, tag, or commit")
    return parser.parse_args()


def fetch_config(model: str, revision: str) -> tuple[dict[str, Any], str]:
    repo_id = normalize_repo_id(model)
    quoted_repo = "/".join(urllib.parse.quote(part) for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/config.json"

    headers = {"User-Agent": "torch-visualiser/tviz"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise RuntimeError(
                f"could not access {repo_id} config.json ({error.code}); set HF_TOKEN for gated/private repos"
            ) from error
        raise RuntimeError(f"could not fetch {repo_id} config.json ({error.code})") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"network error while fetching {repo_id}: {error.reason}") from error

    return json.loads(raw), url


def normalize_repo_id(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith("https://huggingface.co/"):
        path = urllib.parse.urlparse(value).path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        if len(parts) == 1 and parts[0]:
            return parts[0]
    if value.startswith("huggingface.co/"):
        return normalize_repo_id(f"https://{value}")
    return value


def build_ir(model: str, source: str, config: dict[str, Any]) -> dict[str, Any]:
    repo_id = normalize_repo_id(model)
    model_type = str(config.get("model_type") or config.get("architectures", ["model"])[0])
    vision_config = first_dict(config, "vision_config", "vision_config_dict")
    text_config = first_dict(config, "text_config", "language_config", "llm_config") or config
    audio_config = first_dict(config, "audio_config", "speech_config")

    if vision_config and text_config is not config:
        ir = multimodal_ir(repo_id, source, model_type, config, text_config, vision_config, audio_config)
    elif looks_vision(model_type, config):
        ir = vision_ir(repo_id, source, model_type, config)
    elif looks_encoder(model_type, config):
        ir = encoder_ir(repo_id, source, model_type, config)
    else:
        ir = decoder_ir(repo_id, source, model_type, config)

    ir["warnings"] = hf_warnings(config, ir)
    return ir


def decoder_ir(repo_id: str, source: str, model_type: str, config: dict[str, Any]) -> dict[str, Any]:
    hidden = get_int(config, "hidden_size", "n_embd", "d_model", default=4096)
    layers = get_int(config, "num_hidden_layers", "n_layer", "num_layers", default=1)
    heads = get_int(config, "num_attention_heads", "n_head", default=0)
    kv_heads = get_int(config, "num_key_value_heads", "num_kv_heads", default=heads)
    intermediate = get_int(config, "intermediate_size", "ffn_dim", "n_inner", default=hidden * 4)
    vocab = get_int(config, "vocab_size", default=0)
    context = get_int(config, "max_position_embeddings", "n_positions", "max_sequence_length", default=0)
    sliding = get_int(config, "sliding_window", default=0)
    layer_params = estimate_decoder_layer_params(hidden, intermediate)
    total = estimate_total_params(vocab, hidden, layers, layer_params)

    nodes = [
        node("input_ids", "Input IDs", "Input", "input_ids", 0, [], ["batchxtokens"], {"context": context}, "input"),
        node("tok_embeddings", "Token embeddings", "Embedding", "model.embed_tokens", vocab * hidden, ["batchxtokens"], [f"batchxtokensx{hidden}"], {"vocab": vocab}, "embedding"),
        node("rotary", "Rotary / position", "Position", "model.rotary_emb", 0, ["positions"], [f"batchxtokensx{hidden}"], {"context": context}, "embedding"),
        node("decoder_layer", "DecoderLayer", "Module", "model.layers.*", layer_params, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"layers": layers, "hidden": hidden}, "attention", repeated=layers),
        node("attention_norm", "RMS/LayerNorm", "Norm", "model.layers.*.input_norm", hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {}, "norm", repeated=layers, depth=1),
        node("self_attention", "Self attention", "Attention", "model.layers.*.self_attn", hidden * hidden * 4, ["hidden", "rope"], [f"batchxtokensx{hidden}"], {"heads": heads, "KV": kv_heads, "window": sliding}, "attention", repeated=layers, depth=1),
        node("add_1", "Add", "ResidualAdd", "model.layers.*.resid_1", 0, ["hidden", "attention"], [f"batchxtokensx{hidden}"], {"skip": "attention"}, "custom", repeated=layers, depth=1),
        node("mlp_norm", "RMS/LayerNorm", "Norm", "model.layers.*.post_norm", hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {}, "norm", repeated=layers, depth=1),
        node("mlp", "MLP", "MLP", "model.layers.*.mlp", hidden * intermediate * 3, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"intermediate": intermediate}, "mlp", repeated=layers, depth=1),
        node("add_2", "Add", "ResidualAdd", "model.layers.*.resid_2", 0, ["hidden", "mlp"], [f"batchxtokensx{hidden}"], {"skip": "mlp"}, "custom", repeated=layers, depth=1),
        node("final_norm", "Final norm", "Norm", "model.norm", hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {}, "norm"),
        node("lm_head", "LM Head", "Linear", "lm_head", vocab * hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{vocab or 'vocab'}"], {"tie": str(config.get("tie_word_embeddings", False))}, "output"),
    ]
    edges = [
        edge("input_ids", "tok_embeddings"),
        edge("tok_embeddings", "rotary"),
        edge("rotary", "decoder_layer"),
        edge("rotary", "self_attention", "branch"),
        edge("decoder_layer", "attention_norm"),
        edge("attention_norm", "self_attention"),
        edge("decoder_layer", "add_1", "skip"),
        edge("self_attention", "add_1"),
        edge("add_1", "mlp_norm"),
        edge("mlp_norm", "mlp"),
        edge("add_1", "add_2", "skip"),
        edge("mlp", "add_2"),
        edge("add_2", "final_norm"),
        edge("final_norm", "lm_head"),
    ]
    return ir(repo_id, source, total, [input_info("input_ids", "int64", ["batch", "tokens"])], nodes, edges, [
        group("decoder", "Decoder stack", ["decoder_layer", "attention_norm", "self_attention", "add_1", "mlp_norm", "mlp", "add_2"], layers)
    ])


def encoder_ir(repo_id: str, source: str, model_type: str, config: dict[str, Any]) -> dict[str, Any]:
    hidden = get_int(config, "hidden_size", "d_model", default=768)
    layers = get_int(config, "num_hidden_layers", "num_layers", default=1)
    heads = get_int(config, "num_attention_heads", default=0)
    intermediate = get_int(config, "intermediate_size", "encoder_ffn_dim", default=hidden * 4)
    vocab = get_int(config, "vocab_size", default=0)
    layer_params = estimate_decoder_layer_params(hidden, intermediate)
    total = estimate_total_params(vocab, hidden, layers, layer_params)
    nodes = [
        node("input_ids", "Input IDs", "Input", "input_ids", 0, [], ["batchxtokens"], {}, "input"),
        node("embeddings", "Token + position embeddings", "Embedding", "embeddings", vocab * hidden, ["batchxtokens"], [f"batchxtokensx{hidden}"], {"vocab": vocab}, "embedding"),
        node("encoder_layer", "EncoderLayer", "Module", "encoder.layers.*", layer_params, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"layers": layers}, "attention", repeated=layers),
        node("self_attention", "Self attention", "Attention", "encoder.layers.*.attention", hidden * hidden * 4, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"heads": heads}, "attention", repeated=layers, depth=1),
        node("feed_forward", "Feed-forward", "MLP", "encoder.layers.*.mlp", hidden * intermediate * 2, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"intermediate": intermediate}, "mlp", repeated=layers, depth=1),
        node("pooler", "Pooler / head", "Output", "pooler", hidden * hidden, [f"batchxtokensx{hidden}"], [f"batchx{hidden}"], {}, "output"),
    ]
    edges = [
        edge("input_ids", "embeddings"),
        edge("embeddings", "encoder_layer"),
        edge("encoder_layer", "self_attention"),
        edge("self_attention", "feed_forward"),
        edge("feed_forward", "pooler"),
    ]
    return ir(repo_id, source, total, [input_info("input_ids", "int64", ["batch", "tokens"])], nodes, edges, [
        group("encoder", "Encoder stack", ["encoder_layer", "self_attention", "feed_forward"], layers)
    ])


def vision_ir(repo_id: str, source: str, model_type: str, config: dict[str, Any]) -> dict[str, Any]:
    hidden = get_int(config, "hidden_size", "vision_embed_dim", default=768)
    layers = get_int(config, "num_hidden_layers", "num_layers", "depth", default=1)
    heads = get_int(config, "num_attention_heads", "num_heads", default=0)
    image_size = get_int(config, "image_size", default=224)
    patch = get_int(config, "patch_size", default=16)
    intermediate = get_int(config, "intermediate_size", "mlp_dim", default=hidden * 4)
    patches = (image_size // patch) ** 2 if image_size and patch else 0
    layer_params = estimate_decoder_layer_params(hidden, intermediate)
    qkv_params = hidden * hidden
    output_params = hidden * hidden
    mlp_in_params = hidden * intermediate
    mlp_out_params = intermediate * hidden
    seq = f"batchx{patches}x{hidden}"
    nodes = [
        node("image", "Image input", "Input", "pixel_values", 0, [], [f"batchx3x{image_size}x{image_size}"], {"patch": patch}, "input"),
        node("patch_embed", "Patch embedding", "Conv/Linear", "embeddings.patch_embedding", 3 * patch * patch * hidden, [f"batchx3x{image_size}x{image_size}"], [f"batchx{patches}x{hidden}"], {"patch": patch}, "conv"),
        node("vit_layer", "ViTLayer", "Module", "encoder.layer.*", layer_params, [seq], [seq], {"layers": layers}, "attention", repeated=layers),
        node("ln_1", "LayerNorm", "LayerNorm", "encoder.layer.*.layernorm_before", hidden * 2, [seq], [seq], {}, "norm", repeated=layers, depth=1),
        node("vit_attention", "ViTAttention", "Module", "encoder.layer.*.attention", hidden * hidden * 4, [seq], [seq], {"heads": heads}, "attention", repeated=layers, depth=1),
        node("vit_self_attention", "ViTSelfAttention", "Module", "encoder.layer.*.attention.attention", hidden * hidden * 3, [seq], [seq], {"heads": heads}, "attention", repeated=layers, depth=2),
        node("query", "Linear(query)", "Linear", "encoder.layer.*.attention.attention.query", qkv_params, [seq], [seq], {}, "mlp", repeated=layers, depth=3),
        node("key", "Linear(key)", "Linear", "encoder.layer.*.attention.attention.key", qkv_params, [seq], [seq], {}, "mlp", repeated=layers, depth=3),
        node("value", "Linear(value)", "Linear", "encoder.layer.*.attention.attention.value", qkv_params, [seq], [seq], {}, "mlp", repeated=layers, depth=3),
        node("attention_scores", "Attention scores", "MatMul/Softmax", "encoder.layer.*.attention.attention.scores", 0, ["query", "key", "value"], [seq], {"heads": heads}, "attention", repeated=layers, depth=3),
        node("vit_self_output", "ViTSelfOutput", "Module", "encoder.layer.*.attention.output", output_params, [seq], [seq], {}, "attention", repeated=layers, depth=2),
        node("self_output_dense", "Linear", "Linear", "encoder.layer.*.attention.output.dense", output_params, [seq], [seq], {}, "mlp", repeated=layers, depth=3),
        node("self_output_dropout", "Dropout", "Dropout", "encoder.layer.*.attention.output.dropout", 0, [seq], [seq], {}, "activation", repeated=layers, depth=3),
        node("add_1", "Add", "ResidualAdd", "encoder.layer.*.resid_1", 0, ["input", "attention"], [seq], {"skip": "attention"}, "custom", repeated=layers, depth=1),
        node("ln_2", "LayerNorm", "LayerNorm", "encoder.layer.*.layernorm_after", hidden * 2, [seq], [seq], {}, "norm", repeated=layers, depth=1),
        node("vit_intermediate", "ViTIntermediate", "Module", "encoder.layer.*.intermediate", mlp_in_params, [seq], [f"batchx{patches}x{intermediate}"], {"intermediate": intermediate}, "mlp", repeated=layers, depth=1),
        node("intermediate_dense", "Linear", "Linear", "encoder.layer.*.intermediate.dense", mlp_in_params, [seq], [f"batchx{patches}x{intermediate}"], {}, "mlp", repeated=layers, depth=2),
        node("gelu", "GELU", "GELU", "encoder.layer.*.intermediate.intermediate_act_fn", 0, [f"batchx{patches}x{intermediate}"], [f"batchx{patches}x{intermediate}"], {}, "activation", repeated=layers, depth=2),
        node("vit_output", "ViTOutput", "Module", "encoder.layer.*.output", mlp_out_params, [f"batchx{patches}x{intermediate}"], [seq], {}, "mlp", repeated=layers, depth=1),
        node("output_dense", "Linear", "Linear", "encoder.layer.*.output.dense", mlp_out_params, [f"batchx{patches}x{intermediate}"], [seq], {}, "mlp", repeated=layers, depth=2),
        node("output_dropout", "Dropout", "Dropout", "encoder.layer.*.output.dropout", 0, [seq], [seq], {}, "activation", repeated=layers, depth=2),
        node("add_2", "Add", "ResidualAdd", "encoder.layer.*.resid_2", 0, ["hidden", "mlp"], [seq], {"skip": "mlp"}, "custom", repeated=layers, depth=1),
        node("head", "Pool / classifier", "Output", "head", hidden * hidden, [seq], [f"batchx{hidden}"], {}, "output"),
    ]
    edges = [
        edge("image", "patch_embed"),
        edge("patch_embed", "vit_layer"),
        edge("vit_layer", "ln_1"),
        edge("ln_1", "vit_attention"),
        edge("vit_attention", "vit_self_attention"),
        edge("vit_self_attention", "query", "branch"),
        edge("vit_self_attention", "key", "branch"),
        edge("vit_self_attention", "value", "branch"),
        edge("query", "attention_scores", "join"),
        edge("key", "attention_scores", "join"),
        edge("value", "attention_scores", "join"),
        edge("attention_scores", "vit_self_output"),
        edge("vit_self_output", "self_output_dense"),
        edge("self_output_dense", "self_output_dropout"),
        edge("vit_self_attention", "vit_self_output", "summary"),
        edge("vit_attention", "add_1", "summary"),
        edge("vit_self_output", "add_1", "summary"),
        edge("vit_layer", "add_1", "skip"),
        edge("self_output_dropout", "add_1"),
        edge("add_1", "ln_2"),
        edge("ln_2", "vit_intermediate"),
        edge("vit_intermediate", "intermediate_dense"),
        edge("intermediate_dense", "gelu"),
        edge("gelu", "vit_output"),
        edge("vit_intermediate", "vit_output", "summary"),
        edge("vit_output", "output_dense"),
        edge("output_dense", "output_dropout"),
        edge("vit_output", "add_2", "summary"),
        edge("add_1", "add_2", "skip"),
        edge("output_dropout", "add_2"),
        edge("add_2", "head"),
    ]
    return ir(repo_id, source, layers * layer_params, [input_info("pixel_values", "float32", ["batch", "3", str(image_size), str(image_size)])], nodes, edges, [
        group("vision", "Vision encoder", ["vit_layer", "ln_1", "vit_attention", "add_1", "ln_2", "vit_intermediate", "vit_output", "add_2"], layers),
        group("attention", "ViTAttention", ["vit_attention", "vit_self_attention", "vit_self_output"], layers),
        group("self_attention", "ViTSelfAttention", ["vit_self_attention", "query", "key", "value", "attention_scores"], layers),
        group("self_output", "ViTSelfOutput", ["vit_self_output", "self_output_dense", "self_output_dropout"], layers),
        group("intermediate", "ViTIntermediate", ["vit_intermediate", "intermediate_dense", "gelu"], layers),
        group("output", "ViTOutput", ["vit_output", "output_dense", "output_dropout"], layers),
    ])


def multimodal_ir(repo_id: str, source: str, model_type: str, root: dict[str, Any], text: dict[str, Any], vision: dict[str, Any], audio: dict[str, Any] | None) -> dict[str, Any]:
    hidden = get_int(text, "hidden_size", "n_embd", "d_model", default=get_int(root, "hidden_size", default=4096))
    layers = get_int(text, "num_hidden_layers", "n_layer", "num_layers", default=1)
    heads = get_int(text, "num_attention_heads", "n_head", default=0)
    kv_heads = get_int(text, "num_key_value_heads", "num_kv_heads", default=heads)
    intermediate = get_int(text, "intermediate_size", "ffn_dim", default=hidden * 4)
    vocab = get_int(text, "vocab_size", default=get_int(root, "vocab_size", default=0))
    v_hidden = get_int(vision, "hidden_size", "embed_dim", "vision_embed_dim", default=1152)
    v_layers = get_int(vision, "num_hidden_layers", "num_layers", "depth", default=0)
    patch = get_int(vision, "patch_size", default=16)
    image_size = get_int(vision, "image_size", default=224)
    patches = (image_size // patch) ** 2 if image_size and patch else 0
    layer_params = estimate_decoder_layer_params(hidden, intermediate)
    total = estimate_total_params(vocab, hidden, layers, layer_params) + v_layers * estimate_decoder_layer_params(v_hidden, v_hidden * 4)

    nodes = [
        node("image_input", "Image input", "Input", "pixel_values", 0, [], [f"batchx3x{image_size}x{image_size}"], {"patch": patch}, "input"),
        node("text_input", "Text input", "Input", "input_ids", 0, [], ["batchxtokens"], {"vocab": vocab}, "input"),
        node("vision_encoder", "VisionEncoder", "Module", "vision_model", v_layers * estimate_decoder_layer_params(v_hidden, v_hidden * 4), [f"batchx3x{image_size}x{image_size}"], [f"batchx{patches}x{v_hidden}"], {"layers": v_layers, "hidden": v_hidden, "patch": patch}, "conv"),
        node("vision_projection", "Vision projection", "Linear", "multi_modal_projector", v_hidden * hidden, [f"batchx{patches}x{v_hidden}"], [f"batchx{patches}x{hidden}"], {str(v_hidden): hidden}, "mlp"),
        node("text_embeddings", "Text embeddings", "Embedding", "language_model.embed_tokens", vocab * hidden, ["batchxtokens"], [f"batchxtokensx{hidden}"], {"vocab": vocab}, "embedding"),
        node("merge", "masked_scatter", "Join", "merge.masked_scatter", 0, ["vision", "text"], [f"batchxtokensx{hidden}"], {"join": "image tokens into text stream"}, "custom"),
        node("rotary", "Rotary / position", "Position", "language_model.rotary_emb", 0, ["positions"], [f"batchxtokensx{hidden}"], {}, "embedding"),
        node("decoder_layer", "DecoderLayer", "Module", "language_model.layers.*", layer_params, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"layers": layers}, "attention", repeated=layers),
        node("attention_norm", "RMS/LayerNorm", "Norm", "language_model.layers.*.input_norm", hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {}, "norm", repeated=layers, depth=1),
        node("self_attention", "TextAttention", "Attention", "language_model.layers.*.self_attn", hidden * hidden * 4, ["hidden", "rope"], [f"batchxtokensx{hidden}"], {"heads": heads, "KV": kv_heads}, "attention", repeated=layers, depth=1),
        node("add_1", "Add", "ResidualAdd", "language_model.layers.*.resid_1", 0, ["hidden", "attention"], [f"batchxtokensx{hidden}"], {"skip": "attention"}, "custom", repeated=layers, depth=1),
        node("mlp_norm", "RMS/LayerNorm", "Norm", "language_model.layers.*.post_norm", hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {}, "norm", repeated=layers, depth=1),
        node("mlp", "TextMLP", "MLP", "language_model.layers.*.mlp", hidden * intermediate * 3, [f"batchxtokensx{hidden}"], [f"batchxtokensx{hidden}"], {"intermediate": intermediate}, "mlp", repeated=layers, depth=1),
        node("add_2", "Add", "ResidualAdd", "language_model.layers.*.resid_2", 0, ["hidden", "mlp"], [f"batchxtokensx{hidden}"], {"skip": "mlp"}, "custom", repeated=layers, depth=1),
        node("lm_head", "LM Head", "Linear", "lm_head", vocab * hidden, [f"batchxtokensx{hidden}"], [f"batchxtokensx{vocab or 'vocab'}"], {"output": "logits"}, "output"),
    ]
    edges = [
        edge("image_input", "vision_encoder"),
        edge("vision_encoder", "vision_projection"),
        edge("text_input", "text_embeddings"),
        edge("vision_projection", "merge", "join"),
        edge("text_embeddings", "merge", "join"),
        edge("merge", "rotary"),
        edge("rotary", "decoder_layer"),
        edge("rotary", "self_attention", "branch"),
        edge("decoder_layer", "attention_norm"),
        edge("attention_norm", "self_attention"),
        edge("decoder_layer", "add_1", "skip"),
        edge("self_attention", "add_1"),
        edge("add_1", "mlp_norm"),
        edge("mlp_norm", "mlp"),
        edge("add_1", "add_2", "skip"),
        edge("mlp", "add_2"),
        edge("add_2", "lm_head"),
    ]
    inputs = [
        input_info("pixel_values", "float32", ["batch", "3", str(image_size), str(image_size)]),
        input_info("input_ids", "int64", ["batch", "tokens"]),
    ]
    groups = [
        group("modalities", "Parallel modality paths", ["image_input", "text_input", "vision_encoder", "text_embeddings"], 1),
        group("decoder", "Decoder stack", ["decoder_layer", "attention_norm", "self_attention", "add_1", "mlp_norm", "mlp", "add_2"], layers),
    ]
    return ir(repo_id, source, total, inputs, nodes, edges, groups)


def node(id_: str, label: str, kind: str, path: str, params: int, inputs: list[str], outputs: list[str], attrs: dict[str, Any], style: str, repeated: int = 1, depth: int = 0) -> dict[str, Any]:
    return {
        "id": id_,
        "label": label,
        "kind": kind,
        "module_path": path,
        "params": max(0, int(params or 0)),
        "trainable_params": max(0, int(params or 0)),
        "input_shapes": [str(item) for item in inputs],
        "output_shapes": [str(item) for item in outputs],
        "attributes": {str(key): compact(value) for key, value in attrs.items() if value not in {None, 0, ""}},
        "style": style,
        "repeated": max(1, int(repeated or 1)),
        "depth": depth,
    }


def edge(source: str, target: str, kind: str = "data") -> dict[str, str]:
    return {"from": source, "to": target, "kind": kind}


def group(id_: str, label: str, children: list[str], repeated: int = 1) -> dict[str, Any]:
    return {"id": id_, "label": label, "children": children, "repeated": max(1, int(repeated or 1))}


def input_info(name: str, dtype: str, shape: list[str]) -> dict[str, Any]:
    return {"name": name, "dtype": dtype, "shape": [str(item) for item in shape]}


def ir(name: str, source: str, total: int, inputs: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, str]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "model": {"name": name, "source": source, "total_params": max(0, int(total or 0)), "trainable_params": max(0, int(total or 0))},
        "inputs": inputs,
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "warnings": [],
    }


def first_dict(config: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return None


def get_int(config: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return default


def estimate_decoder_layer_params(hidden: int, intermediate: int) -> int:
    return max(0, hidden * hidden * 4 + hidden * intermediate * 3)


def estimate_total_params(vocab: int, hidden: int, layers: int, layer_params: int) -> int:
    return max(0, vocab * hidden + layers * layer_params)


def looks_vision(model_type: str, config: dict[str, Any]) -> bool:
    lowered = model_type.lower()
    return any(token in lowered for token in ["vit", "clip_vision", "swin", "beit", "deit"]) or "image_size" in config


def looks_encoder(model_type: str, config: dict[str, Any]) -> bool:
    lowered = model_type.lower()
    if any(token in lowered for token in ["bert", "roberta", "deberta", "electra", "encoder"]):
        return True
    architectures = " ".join(str(item).lower() for item in config.get("architectures", []))
    return "maskedlm" in architectures or "sequenceclassification" in architectures


def hf_warnings(config: dict[str, Any], ir_payload: dict[str, Any]) -> list[str]:
    warnings = [
        "HF view is config-derived; exact custom forward paths require loading/tracing local model code.",
        "Parameter counts are estimated from config fields when no explicit count exists.",
    ]
    if config.get("trust_remote_code"):
        warnings.append("Config advertises custom code; tviz did not execute remote code.")
    return warnings


def compact(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
