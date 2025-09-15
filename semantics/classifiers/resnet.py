from __future__ import annotations
from typing import Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as tvm

# Helpful: map short names to constructors & default weights enums
# (Works with torchvision >= 0.13; uses .Weights for pretrained)
_RESNET_SPECS = {
    "resnet18":      (tvm.resnet18,      getattr(tvm, "ResNet18_Weights", None)),
    "resnet34":      (tvm.resnet34,      getattr(tvm, "ResNet34_Weights", None)),
    "resnet50":      (tvm.resnet50,      getattr(tvm, "ResNet50_Weights", None)),
    "resnet101":     (tvm.resnet101,     getattr(tvm, "ResNet101_Weights", None)),
    "resnet152":     (tvm.resnet152,     getattr(tvm, "ResNet152_Weights", None)),
    "resnext50_32x4d":   (tvm.resnext50_32x4d,   getattr(tvm, "ResNeXt50_32X4D_Weights", None)),
    "resnext101_32x8d":  (tvm.resnext101_32x8d,  getattr(tvm, "ResNeXt101_32X8D_Weights", None)),
    "wide_resnet50_2":   (tvm.wide_resnet50_2,   getattr(tvm, "Wide_ResNet50_2_Weights", None)),
    "wide_resnet101_2":  (tvm.wide_resnet101_2,  getattr(tvm, "Wide_ResNet101_2_Weights", None)),
}

def _resolve_weights_enum(enum_cls, pretrained: bool):
    """
    Convert a bool 'pretrained' into the recommended default weights enum,
    or None if not available / desired.
    """
    if not pretrained or enum_cls is None:
        return None
    # All modern torchvision Weights enums have DEFAULT
    return enum_cls.DEFAULT


class TorchvisionResNetClassifier(nn.Module):
    """
    Thin wrapper around torchvision ResNet models for classification.

    Features:
      - Choose any torchvision ResNet variant via `arch`
      - Set `num_classes`
      - Support arbitrary `in_channels`
      - Optionally freeze backbone and train only the final head
      - Optionally return penultimate features for downstream tasks

    Args:
      arch: one of keys in _RESNET_SPECS (e.g., "resnet50")
      num_classes: number of classes for the final linear layer
      in_channels: input channels (default 3). If != 3, conv1 is adapted.
      pretrained: load torchvision recommended weights if True.
      freeze_backbone: if True, backbone (all except final fc) is frozen.
      dropout: optional dropout before the final linear layer.
      return_features: if True, forward returns (logits, features)
      global_pool: 'avg' (default), 'max', or 'avgmax' (concat of both)

    Forward:
      x -> logits  (or (logits, features) if return_features=True)

    """
    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 1000,
        in_channels: int = 3,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        return_features: bool = False,
        global_pool: Literal["avg", "max", "avgmax"] = "avg",
    ):
        super().__init__()
        if arch not in _RESNET_SPECS:
            raise ValueError(f"Unknown arch '{arch}'. Valid: {list(_RESNET_SPECS.keys())}")

        ctor, weights_enum = _RESNET_SPECS[arch]
        weights = _resolve_weights_enum(weights_enum, pretrained)

        # Build the torchvision model
        self.backbone = ctor(weights=weights)
        self.arch = arch
        self.return_features = return_features
        self.global_pool = global_pool

        # Adapt first conv if needed
        if in_channels != 3:
            self._replace_conv1(in_channels)

        # Grab the final feature dimension (in_features of .fc)
        if not hasattr(self.backbone, "fc") or not hasattr(self.backbone.fc, "in_features"):
            raise RuntimeError(f"Unexpected ResNet structure for arch '{arch}': missing .fc")
        feat_dim = self.backbone.fc.in_features

        # Replace the classifier head with (optional) dropout + new Linear
        head_layers = []
        if dropout and dropout > 0:
            head_layers.append(nn.Dropout(p=dropout))
        # Note: we'll handle pooling ourselves in forward if global_pool != 'avg'
        # but keep backbone.fc as identity to bypass double pooling.
        self.backbone.fc = nn.Identity()

        # Final classification layer (after pooling)
        self.classifier = nn.Linear(
            feat_dim if global_pool in ("avg", "max") else feat_dim * 2,
            num_classes,
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Re-enable gradients for classifier
            for p in self.classifier.parameters():
                p.requires_grad = True

        # Optional extra dropout just before classifier
        self.head_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _replace_conv1(self, in_channels: int):
        """
        Replace the first convolution to accept arbitrary input channels.
        Strategy:
          - same kernel/stride/padding as original
          - if expanding from 3->C, repeat/average weights; if shrinking, average across RGB
          - if pretrained not available, falls back to Kaiming init
        """
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                w = old_conv.weight  # [out, 3, k, k]
                if in_channels == 1:
                    new_conv.weight.copy_(w.mean(dim=1, keepdim=True))  # grayscale
                elif in_channels > 3:
                    # Repeat channels then truncate or average to match
                    rep = (in_channels + 2) // 3
                    w_rep = w.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
                    new_conv.weight.copy_(w_rep * (3.0 / in_channels))
                else:
                    # in_channels == 2: average RGB into 2 channels with simple scheme
                    avg2 = torch.stack([w[:, :2, :, :].mean(1), w[:, 1:, :, :].mean(1)], dim=1)
                    new_conv.weight[:, :2].copy_(avg2[:, :2])
                    if new_conv.weight.shape[1] > 2:
                        nn.init.kaiming_normal_(new_conv.weight[:, 2:], mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

        self.backbone.conv1 = new_conv

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] after last conv stage
        if self.global_pool == "avg":
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif self.global_pool == "max":
            return F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.global_pool == "avgmax":
            a = F.adaptive_avg_pool2d(x, 1).flatten(1)
            m = F.adaptive_max_pool2d(x, 1).flatten(1)
            return torch.cat([a, m], dim=1)
        else:
            raise ValueError(f"Unknown global_pool '{self.global_pool}'")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          logits  (or (logits, features) if return_features=True)
        """
        # Follow torchvision resnet forward up to the penultimate pooling
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        feats = self._pool(x)               # [B, F] or [B, 2F] for avgmax
        feats = self.head_dropout(feats)
        logits = self.classifier(feats)

        if self.return_features:
            return logits, feats
        return logits

class ResNet18(TorchvisionResNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(arch="resnet18", **kwargs)

class ResNet34(TorchvisionResNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(arch="resnet34", **kwargs)

class ResNet50(TorchvisionResNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(arch="resnet50", **kwargs)

class ResNet101(TorchvisionResNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(arch="resnet101", **kwargs)

class ResNet152(TorchvisionResNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(arch="resnet152", **kwargs)
