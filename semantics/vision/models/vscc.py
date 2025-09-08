import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Variational Source-Channel Coding for Semantic Communication
    Paper: https://arxiv.org/pdf/2410.08222
"""

def _reparameterize(stats: torch.Tensor) -> torch.Tensor:
    mu, logvar = stats.chunk(2, dim=1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class Swish(nn.Module):
    """Swish ≈ SiLU in PyTorch (identical formula)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNetBlock(nn.Module):
    """Residual block: [GN -> Swish -> Conv3x3] x 2 with a skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act1 = Swish()
        self.conv1 = conv3x3(channels, channels)
        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act2 = Swish()
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class Downsample(nn.Module):
    """2x downsample using stride-2 conv that also changes channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x upsample with nearest-neighbor + conv to adjust channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Simple spatial self-attention (dot-product) for 2D feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.q = conv3x3(channels, channels)
        self.k = conv3x3(channels, channels)
        self.v = conv3x3(channels, channels)
        self.proj = conv3x3(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        k = self.k(h).reshape(B, C, H * W)                    # (B, C, HW)
        v = self.v(h).reshape(B, C, H * W).transpose(1, 2)    # (B, HW, C)

        attn = torch.matmul(q, k) * self.scale                # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)                           # (B, HW, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return out + x


class VSCCEncoder(nn.Module):
    """VSCC Encoder"""
    def __init__(
            self, 
            in_ch: int = 3, 
            k: int = 128,
            reparameterize: bool = False
        ):
        super().__init__()
        self.stem = conv3x3(in_ch, 32)

        # Pyramid to 192 channels
        self.block32 = ResNetBlock(32)
        self.down32_64 = Downsample(32, 64)
        self.block64 = ResNetBlock(64)
        self.down64_128 = Downsample(64, 128)
        self.block128 = ResNetBlock(128)
        self.down128_192 = Downsample(128, 192)

        # Bottleneck at 192
        self.mid = nn.Sequential(
            ResNetBlock(192),
            ResNetBlock(192),
            ResNetBlock(192),
            ResNetBlock(192),
            AttentionBlock(192),
            ResNetBlock(192),
        )

        # Heads to parameterize latent (single head emitting [mu | logvar])
        self.pre_head_norm = nn.GroupNorm(32, 192)
        self.pre_head_act = Swish()
        self.stats_head = conv3x3(192, 2 * k)  # [mu | logvar] concatenated along C

        # If True, perform reparameterization in the encoder before transmission through the channel.
        self.reparameterize = reparameterize

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.block32(x)
        x = self.down32_64(x)
        x = self.block64(x)
        x = self.down64_128(x)
        x = self.block128(x)
        x = self.down128_192(x)
        x = self.mid(x)

        h = self.pre_head_act(self.pre_head_norm(x))
        stats = self.stats_head(h)  # (B, 2k, H', W')
        mu, logvar = stats.chunk(2, dim=1)

        if self.reparameterize:
            # Reparameterize before sending through the channel
            z = _reparameterize(stats)
            return z, mu, logvar
        else:
            return stats, mu, logvar


class VSCCDecoder(nn.Module):
    """VSCC Decoder"""
    def __init__(
            self, 
            out_ch: int = 3, 
            k: int = 128,
            reparameterize: bool = True
        ):
        super().__init__()
        self.in_proj = conv3x3(k, 192)

        self.mid = nn.Sequential(
            ResNetBlock(192),
            AttentionBlock(192),
            ResNetBlock(192),
            ResNetBlock(192),
            ResNetBlock(192),
        )

        # Upsample path 192 -> 128 -> 64 -> 32
        self.up192_128 = Upsample(192, 128)
        self.block128 = ResNetBlock(128)
        self.up128_64 = Upsample(128, 64)
        self.block64 = ResNetBlock(64)
        self.up64_32 = Upsample(64, 32)
        self.block32a = ResNetBlock(32)
        self.block32b = ResNetBlock(32)

        self.tail_norm = nn.GroupNorm(32, 32)
        self.tail_act = Swish()
        self.out_conv = conv3x3(32, out_ch)
        self.out_tanh = nn.Tanh()

        # If True, perform reparameterization in the decoder after transmission through the channel.
        self.reparameterize = reparameterize

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if self.reparameterize:
            # Reparameterize after receiving from the channel
            z = _reparameterize(stats)
        else:
            z = stats
        x = self.in_proj(z)
        x = self.mid(x)
        x = self.up192_128(x)
        x = self.block128(x)
        x = self.up128_64(x)
        x = self.block64(x)
        x = self.up64_32(x)
        x = self.block32a(x)
        x = self.block32b(x)
        x = self.out_conv(self.tail_act(self.tail_norm(x)))
        x = self.out_tanh(x)
        return x, z, stats


class Normalize(nn.Module):
    def __init__(self, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Denormalize(Normalize):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class VSCC(nn.Module):
    """Variational Source-Channel Coding Model"""
    def __init__(self, img_channels: int = 3, latent_k: int = 128,
                 norm_stats: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None):
        super().__init__()
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        if norm_stats is not None:
            mean, std = norm_stats

        self.normalize = Normalize(mean, std)
        self.denormalize = Denormalize(mean, std)

        self.encoder = VSCCEncoder(in_ch=img_channels, k=latent_k)
        self.decoder = VSCCDecoder(out_ch=img_channels, k=latent_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Normalization (as in the figure)
        x_n = self.normalize(x)

        # Encode -> stats = concat(mu, logvar)
        stats, mu, logvar = self.encoder(x_n)

        # Reparameterize
        z = self.encoder._reparameterize(mu, logvar)

        # Decode + Denormalize
        y_n = self.decoder(z)
        y = self.denormalize(y_n)

        aux = {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "stats": stats
        }
        return y, aux