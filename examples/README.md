# Sample Models

This directory contains local PyTorch models for testing `torch-visualiser`.
They are intentionally small enough to run on CPU, while still exercising model
structures that matter for visualization.

## Available Factories

- `tiny_convnet`: CNN with a stem, repeated residual blocks, pooling, and a
  classifier.
- `residual_mlp`: MLP with repeated residual feed-forward blocks.
- `branching_cnn`: CNN with two parallel branches that join by concatenation.
- `mini_transformer`: compact decoder-only transformer with embeddings,
  repeated attention/MLP blocks, residual paths, and an LM head.
- `messy_research_model`: Lightning-shaped research model with multi-input
  data, `ModuleDict` branches, metadata fusion, routing weights, auxiliary
  heads, and training/validation pipeline methods.

## Manual Smoke Test

From the repo root:

```bash
python -m pip install torch
python examples/models.py
```

Expected future `tviz` usage:

```bash
tviz examples/models.py --factory tiny_convnet --input "float32[1,3,64,64]"
tviz examples/models.py --factory residual_mlp --input "float32[8,128]"
tviz examples/models.py --factory branching_cnn --input "float32[1,3,64,64]"
tviz examples/models.py --factory mini_transformer --input "int64[2,16]"
tviz examples/models.py --factory messy_research_model
```

You can test the Rust renderer without PyTorch by using the fixture IR:

```bash
cargo run -p tviz-cli -- fixtures/tiny_convnet.json --no-color
```
