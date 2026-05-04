# torch-visualiser Architecture Plan

`torch-visualiser` is a terminal-first architecture viewer for PyTorch models.
The CLI command is `tviz`; the rendering engine is Rust; the model
introspection layer is Python because arbitrary `torch.nn.Module` definitions
are Python programs.

## Product Goal

The terminal output must feel like a real architecture viewer, not a Markdown
dump or a plain tree of lines. The viewer should be useful as a reference while
reading code, debugging shapes, comparing model variants, or inspecting a
checkpoint before training.

The default experience should answer:

- What are the main blocks?
- How does data flow through them?
- Which blocks repeat?
- Where are residuals, branches, joins, and modality-specific paths?
- What are the tensor shapes and parameter counts?
- Which module is selected, and what are its details?

## High-Level Pipeline

```text
PyTorch model / checkpoint / Hugging Face repo
        |
        v
Python probe
        |
        v
ModelIR JSON
        |
        v
Rust layout engine
        |
        v
Rich terminal renderer
```

The Python probe emits a stable intermediate representation. Rust does not
need to import PyTorch or execute user model code directly.

## CLI Shape

Planned examples:

```bash
tviz examples/models.py --factory tiny_convnet --input "float32[1,3,64,64]"
tviz examples/models.py --factory residual_mlp --input "float32[8,128]"
tviz examples/models.py --factory branching_cnn --input "float32[1,3,64,64]"
tviz examples/models.py --factory mini_transformer --input "int64[2,16]"
tviz Qwen/Qwen3.5-0.8B
tviz https://huggingface.co/google/vit-base-patch16-224
```

For local Python, the user explicitly points at code and a factory. For unsafe
formats, the CLI should require explicit trust flags.

## ModelIR

The Python probe should produce JSON with this broad shape:

```json
{
  "schema_version": "0.1",
  "model": {
    "name": "TinyConvNet",
    "source": "examples/models.py",
    "total_params": 124650,
    "trainable_params": 124650
  },
  "inputs": [
    { "name": "x", "dtype": "float32", "shape": [1, 3, 64, 64] }
  ],
  "nodes": [
    {
      "id": "stem.0",
      "label": "Conv2d",
      "kind": "module",
      "module_path": "stem.0",
      "params": 864,
      "input_shapes": ["1x3x64x64"],
      "output_shapes": ["1x24x32x32"],
      "style": "conv"
    }
  ],
  "edges": [
    { "from": "stem.0", "to": "stem.1", "kind": "data" }
  ],
  "groups": [
    {
      "id": "blocks",
      "label": "ResidualBlock x3",
      "children": ["blocks.0", "blocks.1", "blocks.2"]
    }
  ],
  "warnings": []
}
```

## Introspection Modes

### Structure Mode

Uses `named_modules()`, `named_parameters()`, `named_buffers()`, and
`state_dict()`. This mode does not run the model and should work for the widest
range of custom modules.

### Hook Trace Mode

Runs one forward pass using sample inputs. Forward hooks record execution order,
input shapes, output shapes, dtypes, devices, and module timings. This should be
the default rich mode for local custom models.

### FX Mode

Uses `torch.fx` when available to build a dataflow graph. This is valuable for
branches, joins, function calls, and operator-level details, but it will not
cover every dynamic Python model.

### Export Mode

Uses `torch.export` to obtain a normalized lower-level graph with lifted
parameters and shape metadata. This is an advanced view, useful when the user
wants closer-to-operator behavior.

## Rendering Bar

The renderer should be closer to a terminal-native graph viewer than a textual
summary.

Required visual features:

- Box-drawn modules with distinct styling by kind: conv, norm, activation,
  attention, MLP, embedding, pooling, output, custom.
- Directed edges with arrowheads, including residual skip arcs and branch joins.
- Nested group boxes for sequential containers, residual blocks, attention
  blocks, encoder/decoder stacks, and repeated cycles.
- Repetition compression such as `DecoderLayer x32` and expandable instances.
- Color, emphasis, and compact badges for params, shapes, dtype, trainable
  status, and warnings.
- Stable pan/zoom/granularity behavior so the graph is usable in small
  terminals.
- A detail pane for the selected node with module path, constructor-like
  details, parameter breakdown, input/output shapes, and source hints.
- Search and navigation by module path or type.

Rendering should have tiers:

1. Cell UI: portable Ratatui widgets, Unicode box drawing, styled spans.
2. Dense graph mode: high-density terminal glyphs and braille where useful.
3. Optional pixel mode: Kitty/Sixel snapshots for terminals that support richer
   graphics, inspired by ProteinView's tiered rendering approach.

The first milestone should implement tier 1 well before adding pixel protocols.
Even tier 1 must look designed: color-coded blocks, clean spacing, selected
state, grouped architecture, and readable edges.

## Layout Strategy

Start with a layered directed graph layout:

- Inputs at the top or left, outputs at the bottom or right.
- Sequential paths flow in one dominant direction.
- Branches spread horizontally and rejoin cleanly.
- Residual edges route around groups instead of crossing through labels.
- Repeated blocks collapse into one group by default.

For transformer-like models, prefer a semantic layout over a fully expanded DAG:

```text
token ids
  |
Embedding
  |
DecoderLayer xN
  |-- Attention
  |-- MLP
  |-- residual / norm paths
  |
LM Head
```

For CNNs, emphasize spatial resolution changes and channel width changes.

## Safety Defaults

Loading user Python code is code execution. The CLI should be explicit about
that:

- Local `.py` files require `--factory`.
- Full pickle loading requires `--trust-pickle`.
- Hugging Face `trust_remote_code` requires `--trust-remote-code`.
- Hugging Face config-only mode should avoid downloading model weights.

## Milestones

1. Scaffold Rust CLI, Rust IR types, Python probe package, and examples.
2. Render fixture ModelIR JSON in a rich static TUI.
3. Implement Python structure probe for local model factories.
4. Add hook trace mode and shape capture.
5. Add interaction: pan, zoom, granularity, collapse, search, details.
6. Add Hugging Face config-only architecture adapters.
7. Add FX/export modes.
8. Add optional high-density or pixel render modes.

