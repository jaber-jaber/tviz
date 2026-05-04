"""Small PyTorch models for torch-visualiser development.

Each factory returns an initialized `torch.nn.Module`. The companion
`sample_input_for()` helper returns inputs suitable for a CPU forward pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.main = nn.Sequential(
            ConvBnAct(channels, hidden_channels),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x + self.main(x))


class TinyConvNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct(3, 24, stride=2),
            ConvBnAct(24, 48, stride=2),
        )
        self.blocks = nn.Sequential(
            ConvResidualBlock(48),
            ConvResidualBlock(48),
            ConvResidualBlock(48, hidden_channels=96),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class ResidualFeedForward(nn.Module):
    def __init__(self, width: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ResidualMLP(nn.Module):
    def __init__(self, width: int = 128, depth: int = 6, num_classes: int = 5) -> None:
        super().__init__()
        self.input_proj = nn.Linear(width, width)
        self.blocks = nn.ModuleList(
            [ResidualFeedForward(width, expansion=2) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.norm(x))


class BranchingCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.stem = ConvBnAct(3, 32, stride=2)
        self.small_kernel_branch = nn.Sequential(
            ConvBnAct(32, 32, kernel_size=3),
            ConvBnAct(32, 48, kernel_size=3),
        )
        self.large_kernel_branch = nn.Sequential(
            ConvBnAct(32, 32, kernel_size=5),
            ConvBnAct(32, 48, kernel_size=5),
        )
        self.fusion = nn.Sequential(
            ConvBnAct(96, 64, kernel_size=1),
            ConvResidualBlock(64),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        small = self.small_kernel_branch(x)
        large = self.large_kernel_branch(x)
        x = torch.cat([small, large], dim=1)
        x = self.fusion(x)
        return self.classifier(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.proj(self.dropout(attn))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        width: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_tokens, width)
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, num_heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, tokens = token_ids.shape
        positions = torch.arange(tokens, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.pool(x)).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class MultiScaleStem(nn.Module):
    def __init__(self, out_channels: int = 48) -> None:
        super().__init__()
        branch_channels = out_channels // 3
        self.branches = nn.ModuleDict(
            {
                "small": ConvBnAct(3, branch_channels, kernel_size=3, stride=2),
                "medium": ConvBnAct(3, branch_channels, kernel_size=5, stride=2),
                "wide": nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    ConvBnAct(3, branch_channels, kernel_size=1),
                ),
            }
        )
        self.mix = ConvBnAct(branch_channels * 3, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = [branch(image) for branch in self.branches.values()]
        return self.mix(torch.cat(features, dim=1))


class MetadataTower(nn.Module):
    def __init__(self, in_features: int = 12, width: int = 48) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(in_features)
        self.path = nn.Sequential(
            nn.Linear(in_features, width),
            nn.GELU(),
            ResidualFeedForward(width, expansion=2, dropout=0.0),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.path(self.normalizer(metadata))


class CrossModalMixer(nn.Module):
    def __init__(self, width: int = 96, num_heads: int = 4) -> None:
        super().__init__()
        self.image_proj = nn.Linear(48, width)
        self.meta_proj = nn.Linear(48, width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = ResidualFeedForward(width, expansion=2, dropout=0.0)

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        pooled = image_features.mean(dim=(-2, -1))
        tokens = torch.stack(
            [self.image_proj(pooled), self.meta_proj(metadata_features)],
            dim=1,
        )
        mixed, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.out(self.norm(mixed)).flatten(1)


class MessyResearchCore(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.image = MultiScaleStem(out_channels=48)
        self.metadata = MetadataTower(in_features=12, width=48)
        self.stage_bank = nn.ModuleDict(
            {
                "local": nn.Sequential(
                    DepthwiseSeparableConv(48),
                    SqueezeExcite(48),
                ),
                "context": nn.Sequential(
                    ConvResidualBlock(48, hidden_channels=96),
                    ConvResidualBlock(48),
                ),
                "repair": nn.Conv2d(48, 48, kernel_size=1),
            }
        )
        self.route_weights = nn.Parameter(torch.tensor([0.45, 0.45, 0.10]))
        self.fusion = CrossModalMixer(width=96, num_heads=4)
        self.heads = nn.ModuleDict(
            {
                "main": nn.Sequential(
                    nn.LayerNorm(192),
                    nn.Linear(192, 96),
                    nn.GELU(),
                    nn.Linear(96, num_classes),
                ),
                "quality": nn.Sequential(
                    nn.LayerNorm(192),
                    nn.Linear(192, 1),
                ),
            }
        )
        self.aux_classifier = nn.Linear(48, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.image(image)
        local = self.stage_bank["local"](image_features)
        context = self.stage_bank["context"](image_features)
        repair = self.stage_bank["repair"](image_features)
        weights = self.route_weights.softmax(dim=0)
        image_features = weights[0] * local + weights[1] * context + weights[2] * repair

        metadata_features = self.metadata(metadata)
        fused = self.fusion(image_features, metadata_features)
        pooled = image_features.mean(dim=(-2, -1))

        return {
            "logits": self.heads["main"](fused),
            "quality": self.heads["quality"](fused),
            "aux_logits": self.aux_classifier(pooled),
        }


class MessyLightningLikeModule(nn.Module):
    """A Lightning-shaped wrapper without depending on Lightning.

    This is intentionally more like production research code than a tutorial
    model: the actual architecture is spread across helper modules, dictionaries,
    auxiliary heads, and pipeline-style methods.
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.network = MessyResearchCore(num_classes=num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network(image, metadata)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self(batch["image"], batch["metadata"])
        main = self.loss(outputs["logits"], batch["target"])
        aux = self.loss(outputs["aux_logits"], batch["target"])
        quality_penalty = outputs["quality"].square().mean()
        return main + 0.25 * aux + 0.01 * quality_penalty

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self(batch["image"], batch["metadata"])
        return {"val_logits": outputs["logits"], "val_quality": outputs["quality"]}

    def configure_optimizers(self) -> str:
        return "AdamW(lr=3e-4, weight_decay=0.05)"


def tiny_convnet() -> nn.Module:
    return TinyConvNet()


def residual_mlp() -> nn.Module:
    return ResidualMLP()


def branching_cnn() -> nn.Module:
    return BranchingCNN()


def mini_transformer() -> nn.Module:
    return MiniTransformerLM()


def messy_research_model() -> nn.Module:
    return MessyLightningLikeModule()


MODEL_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "tiny_convnet": tiny_convnet,
    "residual_mlp": residual_mlp,
    "branching_cnn": branching_cnn,
    "mini_transformer": mini_transformer,
    "messy_research_model": messy_research_model,
}

DEFAULT_MODEL = "tiny_convnet"


def get_model(name: str) -> nn.Module:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown sample model {name!r}; choose one of: {choices}") from error


def sample_input_for(name: str) -> tuple[torch.Tensor, ...]:
    if name == "tiny_convnet":
        return (torch.randn(1, 3, 64, 64),)
    if name == "residual_mlp":
        return (torch.randn(8, 128),)
    if name == "branching_cnn":
        return (torch.randn(1, 3, 64, 64),)
    if name == "mini_transformer":
        return (torch.randint(0, 256, (2, 16), dtype=torch.long),)
    if name == "messy_research_model":
        image = torch.randn(2, 3, 64, 64)
        metadata = torch.randn(2, 12)
        return image, metadata
    choices = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(f"unknown sample model {name!r}; choose one of: {choices}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def describe_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> str:
    if isinstance(output, torch.Tensor):
        return f"{tuple(output.shape)} {output.dtype}"
    return ", ".join(f"{tuple(item.shape)} {item.dtype}" for item in output)


def run_smoke_test(model_name: str) -> None:
    model = get_model(model_name)
    model.eval()
    total, trainable = count_parameters(model)
    inputs = sample_input_for(model_name)
    with torch.no_grad():
        output = model(*inputs)
    input_shapes = ", ".join(str(tuple(item.shape)) for item in inputs)
    print(f"{model_name}: {model.__class__.__name__}")
    print(f"  params: {total:,} total / {trainable:,} trainable")
    print(f"  inputs: {input_shapes}")
    print(f"  output: {describe_output(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test sample PyTorch models.")
    parser.add_argument(
        "model",
        nargs="?",
        choices=sorted(MODEL_REGISTRY),
        help="Run one model. Omit to run all samples.",
    )
    args = parser.parse_args()

    names = [args.model] if args.model else sorted(MODEL_REGISTRY)
    for index, name in enumerate(names):
        if index:
            print()
        run_smoke_test(name)


if __name__ == "__main__":
    main()
