"""
models/convnet.py — QuickDrop ConvNet for CIFAR-10 (Method 5)
=============================================================

A faithful, self-contained copy of the ConvNet architecture used by
QuickDrop (Dhasade et al.) — the network the Method 5 gold-standard and
provider models were trained with. It is reproduced here verbatim so the
dissertation's verification engine can ``load_state_dict`` those trained
weights without importing the upstream QuickDrop codebase.

Default CIFAR-10 setting (``get_default_convnet_setting``):
    net_width=128, net_depth=3, act=ReLU, norm=InstanceNorm, pool=AvgPool.

Key structural note: ``instancenorm`` is implemented as
``nn.GroupNorm(C, C, affine=True)`` (one group per channel). GroupNorm has
ONLY ``weight`` + ``bias`` and NO running buffers, so the state_dict is 14
pure weight/bias entries — there is no BatchNorm-style buffer-key mismatch
risk against the trained weights.

This module is additive. It does not modify the ResNet path; ``build_model``
in ``models/resnet.py`` dispatches to ``build_convnet`` only when the config
architecture is ``"convnet"`` (resnet18 remains the default).

Usage:
    from models.convnet import build_convnet
    model = build_convnet(num_classes=10)        # CPU; caller .to(device)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation (included for verbatim fidelity; unused at default)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvNet(nn.Module):
    """ConvNet matching QuickDrop's architecture (utils/networks.py).

    Reproduced verbatim. With the default CIFAR-10 setting the feature
    extractor is three [Conv2d → GroupNorm → ReLU → AvgPool] blocks,
    followed by a single linear classifier over the flattened features.
    """

    def __init__(
        self,
        channel: int,
        num_classes: int,
        net_width: int,
        net_depth: int,
        net_act: str,
        net_norm: str,
        net_pooling: str,
        im_size: Tuple[int, int] = (32, 32),
    ) -> None:
        super().__init__()
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size,
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → flatten → classifier."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened features (penultimate representation)."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act: str) -> nn.Module:
        if net_act == "sigmoid":
            return nn.Sigmoid()
        elif net_act == "relu":
            return nn.ReLU(inplace=True)
        elif net_act == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == "swish":
            return Swish()
        else:
            raise ValueError(f"unknown activation function: {net_act}")

    def _get_pooling(self, net_pooling: str) -> nn.Module | None:
        if net_pooling == "maxpooling":
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            return None
        else:
            raise ValueError(f"unknown net_pooling: {net_pooling}")

    def _get_normlayer(self, net_norm: str, shape_feat) -> nn.Module | None:
        # shape_feat = (c, h, w)
        if net_norm == "batchnorm":
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layernorm":
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instancenorm":
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "groupnorm":
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            return None
        else:
            raise ValueError(f"unknown net_norm: {net_norm}")

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size,
    ):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [
                nn.Conv2d(
                    in_channels, net_width, kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != "none":
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def get_default_convnet_setting() -> Tuple[int, int, str, str, str]:
    """QuickDrop's default ConvNet hyperparameters (width, depth, act, norm, pool)."""
    return 128, 3, "relu", "instancenorm", "avgpooling"


def build_convnet(
    num_classes: int = 10,
    channel: int = 3,
    im_size: Tuple[int, int] = (32, 32),
) -> ConvNet:
    """Factory — build the default CIFAR-10 ConvNet on CPU.

    Matches ``build_model``'s contract: returns a CPU module with arbitrary
    initialisation; the caller moves it to device and loads trained weights.
    Initialisation is irrelevant for verification because ``load_state_dict``
    overwrites all parameters.
    """
    net_width, net_depth, net_act, net_norm, net_pooling = (
        get_default_convnet_setting()
    )
    return ConvNet(
        channel=channel,
        num_classes=num_classes,
        net_width=net_width,
        net_depth=net_depth,
        net_act=net_act,
        net_norm=net_norm,
        net_pooling=net_pooling,
        im_size=im_size,
    )


# ── Verification ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("convnet.py: running verification\n")

    model = build_convnet(num_classes=10)
    total = sum(p.numel() for p in model.parameters())
    keys = list(model.state_dict().keys())
    print(f"[1/3] Model built: {total:,} parameters, {len(keys)} state_dict keys")
    for k in keys:
        print(f"        {k}")

    dummy = torch.randn(4, 3, 32, 32)
    logits = model(dummy)
    assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"
    print(f"[2/3] Forward pass: {tuple(dummy.shape)} -> {tuple(logits.shape)}")

    # [3/3] DEFINITIVE key-alignment proof against a real trained model.
    # Pass a gold/provider .pt path as argv[1]; load_state_dict must succeed.
    if len(sys.argv) > 1:
        sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)  # raises on any key/shape mismatch
        print(f"[3/3] load_state_dict on {sys.argv[1]}: SUCCESS (keys aligned)")
    else:
        print("[3/3] (skipped) pass a gold/provider .pt path to prove key alignment")

    print("\nconvnet.py: checks passed")
