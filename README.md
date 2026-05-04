# torch-visualiser

`torch-visualiser` is a terminal viewer for PyTorch model architectures. The
CLI command is `tviz`; Rust handles terminal rendering, and a Python probe
handles custom PyTorch model introspection.

## Current MVP

This repo currently includes:

- A Rust workspace with a `tviz` CLI.
- A Rust ModelIR parser and layered terminal DAG renderer.
- A Python probe for local `torch.nn.Module` factories.
- Hugging Face `config.json` integration for repo IDs and model URLs.
- Sample PyTorch models in `examples/models.py`.
- Fixture ModelIR files for renderer testing without PyTorch, including a
  multimodal benchmark with parallel paths, joins, skip edges, and container
  modules.

## Install

From the repo root:

```bash
cargo install --path crates/tviz-cli
```

After that, run the CLI directly:

```bash
tviz fixtures/tiny_convnet.json
tviz fixtures/benchmark_multimodal.json
```

## Try The Renderer

```bash
tviz fixtures/tiny_convnet.json
tviz fixtures/benchmark_multimodal.json
```

Open a one-screen alternate-screen view:

```bash
tviz fixtures/tiny_convnet.json --screen
```

Disable ANSI color:

```bash
tviz fixtures/tiny_convnet.json --no-color
```

## Try A Local PyTorch Model

Install PyTorch first:

```bash
python -m pip install torch
```

Then run:

```bash
tviz examples/models.py
tviz examples/models.py --factory tiny_convnet --input "float32[1,3,64,64]"
tviz examples/models.py --factory branching_cnn --input "float32[1,3,64,64]"
tviz examples/models.py --factory mini_transformer --input "int64[2,16]"
tviz examples/models.py --factory messy_research_model
```

## Try A Hugging Face Model

HF integration fetches `config.json` only; it does not download model weights.

```bash
tviz Qwen/Qwen3.5-0.8B
tviz sshleifer/tiny-gpt2
tviz https://huggingface.co/google/vit-base-patch16-224
```

Use a specific branch, tag, or commit:

```bash
tviz sshleifer/tiny-gpt2 --revision main
```

For gated or private repos:

```bash
HF_TOKEN=... tviz org/private-model
```

For dense research models, default print mode shows the full traced execution.
Use a lower granularity when you want a shorter architectural overview:

```bash
tviz examples/models.py --factory messy_research_model --granularity 1
```

Granularity is progressive:

- `--granularity 0` shows the outer architecture.
- `--granularity 1` opens the main repeated blocks.
- `--granularity 2` opens nested modules inside those blocks.
- `--granularity 3` opens individual operations such as projections, activation, and dropout.
- `--granularity 4` shows everything tviz inferred.

For example, ViT models expose the transformer layer internals at higher granularity:

```bash
tviz https://huggingface.co/google/vit-base-patch16-224 --granularity 4
```

If PyTorch lives in a separate environment, point `tviz` at that Python:

```bash
TVIZ_PYTHON=/path/to/env/bin/python tviz examples/models.py --factory messy_research_model
tviz examples/models.py --python /path/to/env/bin/python --factory messy_research_model
```

For raw ModelIR JSON:

```bash
tviz examples/models.py --factory tiny_convnet --json
```

Export alongside the terminal diagram:

```bash
tviz fixtures/benchmark_multimodal.json --export architecture.dot
tviz fixtures/benchmark_multimodal.json --export architecture.svg
tviz fixtures/benchmark_multimodal.json --export architecture.json
```

For development without installing:

```bash
cargo run -p tviz-cli -- fixtures/tiny_convnet.json
```

## Current Limits

- The renderer is currently a rich static terminal DAG render, not an
  interactive pan/zoom TUI yet. Default output is scrollback-friendly. Use
  `--screen` for an alternate-screen fit view.
- Container modules render as labeled group boxes when their children are
  visible in the current granularity level.
- Local PyTorch probing requires PyTorch in the active Python environment.
  Use `TVIZ_PYTHON` or `--python` when launching from outside that environment.
- `tviz model.py` can work without flags when the file exposes a conventional
  zero-argument factory such as `build_model()`, `create_model()`, `get_model()`,
  `model()`, or a `MODEL_REGISTRY` with `DEFAULT_MODEL`. Without a sample input,
  `tviz` falls back to structure mode.
- Hugging Face support is config-derived. It fetches `config.json` only, does
  not execute remote code, and estimates parameter counts when exact counts are
  unavailable.
- `--export` supports `.json`, `.dot`, and `.svg`. The `.dot` export is the most
  useful graph file today; the `.svg` export embeds the DOT source as a simple
  portable artifact until native SVG layout matures.
- The Rust implementation intentionally avoids external crates for the first
  pass, so CLI parsing and JSON parsing are minimal.
