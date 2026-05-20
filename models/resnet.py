"""
models/resnet.py — ResNet-18 for CIFAR-10
==========================================

Standard ResNet-18 adapted for 32×32 inputs. The two key changes from
the ImageNet variant:

  1. First conv is 3×3 (not 7×7) with stride 1 and padding 1.
  2. The initial max-pool layer is removed entirely.

These are well-established modifications (He et al. 2016 appendix,
widely adopted in FL research) that prevent the 32×32 spatial
dimensions from collapsing too early.

Usage:
    from models.resnet import build_model
    model = build_model(num_classes=10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two-layer residual block with optional downsampling."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """ResNet-18 tailored for CIFAR-10 (32×32 RGB inputs).

    Architecture: conv1 → [layer1 × 2] → [layer2 × 2] → [layer3 × 2]
                  → [layer4 × 2] → global avg pool → FC(num_classes)

    Total parameters: ~11.2M (identical to standard ResNet-18 aside
    from the first conv layer).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64

        # CIFAR adaptation: 3×3 conv, stride 1, no max-pool.
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Four residual layers — channel progression: 64 → 128 → 256 → 512.
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Kaiming initialisation for conv layers, standard for ResNets.
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all ResNet layers."""
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_model(num_classes: int = 10) -> ResNet18CIFAR:
    """Factory function — single entry point for the rest of the codebase.

    All other modules call this instead of instantiating the class
    directly, so swapping architectures later requires changing only
    this function.
    """
    return ResNet18CIFAR(num_classes=num_classes)


# ── Verification ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.seeding import set_global_seed

    print("resnet.py: running verification\n")

    # 1. Build and inspect.
    model = build_model(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1/3] Model built: {total_params:,} parameters ({trainable:,} trainable)")

    # 2. Forward pass with CIFAR-10-shaped input.
    dummy = torch.randn(4, 3, 32, 32)
    logits = model(dummy)
    assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"
    print(f"[2/3] Forward pass: input {tuple(dummy.shape)} → output {tuple(logits.shape)}")

    # 3. Deterministic initialisation.
    set_global_seed(42)
    m1 = build_model()
    set_global_seed(42)
    m2 = build_model()
    w1 = list(m1.parameters())[0].data
    w2 = list(m2.parameters())[0].data
    assert torch.equal(w1, w2), "Same seed must produce same initial weights"
    print("[3/3] Deterministic init verified: same seed → same weights")

    print("\nresnet.py: all checks passed ✓")
