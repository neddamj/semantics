from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, Set

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # timm

"""
    SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding
    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10589474
"""

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None,
                 act_layer=nn.GELU, drop: float=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int,
                 qkv_bias: bool=True, qk_scale: Optional[float]=None,
                 attn_drop: float=0., proj_drop: float=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, add_token: bool=True,
                token_num: int=0, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N: int) -> int:
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    """Downsample by 2x (tokens), linear reduce 4*C -> out_dim."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int, out_dim: Optional[int]=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        out_dim = out_dim or dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "PatchMerging: wrong token length"
        assert H % 2 == 0 and W % 2 == 0, "Input H,W must be even"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)      # B, H/2, W/2, 4C
        x = x.view(B, H * W // 4, 4 * C)             # B, (H*W)/4, 4C
        x = self.norm(x)
        x = self.reduction(x)                        # -> out_dim
        return x

    def flops(self) -> int:
        H, W = self.input_resolution
        return H * W * self.dim + (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim


class PatchReverseMerging(nn.Module):
    """Upsample by 2x (tokens) via PixelShuffle-style linear expand."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int, out_dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "PatchReverseMerging: wrong token length"
        assert H % 2 == 0 and W % 2 == 0, "Input H,W must be even for upsample grid"
        x = self.norm(x)
        x = self.increment(x)                        # B, L, out_dim*4
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, 4*out_dim, H, W
        x = nn.PixelShuffle(2)(x)                    # B, out_dim, 2H, 2W
        x = x.flatten(2).permute(0, 2, 1)            # B, (2H*2W), out_dim
        return x

    def flops(self) -> int:
        H, W = self.input_resolution
        return (H * 2) * (W * 2) * self.dim // 4 + (H * 2) * (W * 2) * (self.dim // 4) * self.dim


class AdaptiveModulator(nn.Module):
    """Simple 1D MLP mapper for SNR/rate scalars -> modulation vector of size M."""
    def __init__(self, M: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid(),
        )

    def forward(self, snr_or_rate: torch.Tensor) -> torch.Tensor:
        return self.fc(snr_or_rate)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int=7, shift_size: int=0,
                 mlp_ratio: float=4., qkv_bias: bool=True, qk_scale: Optional[float]=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1, H, W, 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, ws, ws, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "SwinTransformerBlock: wrong token length"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # windows
        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )

        # attention
        attn_windows = self.attn(x_windows, add_token=False, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def update_mask(self) -> None:
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.to(next(self.parameters()).device)

class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int]=(224, 224), patch_size: int=4,
                 in_chans: int=3, embed_dim: int=96, norm_layer: Optional[nn.Module]=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self) -> int:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class EncoderBasicLayer(nn.Module):
    """One encoder stage: optional downsample, then N Swin blocks at reduced res."""
    def __init__(self, dim: int, out_dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float=4., qkv_bias: bool=True, qk_scale: Optional[float]=None,
                 norm_layer=nn.LayerNorm, downsample: Optional[nn.Module]=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer) \
            if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def update_resolution(self, H: int, W: int) -> None:
        for blk in self.blocks:
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class SwinJSCCEncoder(nn.Module):
    def __init__(
            self, 
            img_size: Tuple[int, int], 
            patch_size: int, 
            in_chans: int,
            embed_dims: Iterable[int], 
            depths: Iterable[int], 
            num_heads: Iterable[int], 
            C: Optional[int],
            window_size: int=4,
            mlp_ratio: float=4., 
            qkv_bias: bool=True, 
            qk_scale: Optional[float]=None,
            norm_layer=nn.LayerNorm, 
            patch_norm: bool=True, 
            bottleneck_dim: int=16, 
            model: str='SwinJSCC_w/o_SAandRA',
            snr: float=10.0,
            rate: int=16
        ):
        super().__init__()
        embed_dims = list(embed_dims)
        depths = list(depths)
        num_heads = list(num_heads)

        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[-1] * 1.5)
        self.layer_num = 7
        self.model = model
        self.snr = snr
        self.rate = rate

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderBasicLayer(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else in_chans,
                out_dim=int(embed_dims[i_layer]),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != 0 else None
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], C) if C is not None else nn.Identity()

        # SNR/Rate adaptive subnets
        self.bm_list = nn.ModuleList([AdaptiveModulator(self.hidden_dim) for _ in range(self.layer_num)])
        self.sm_list = nn.ModuleList([nn.Linear(self.embed_dims[-1] if i == 0 else self.hidden_dim,
                                                self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim)
                                      for i in range(self.layer_num)])
        self.sigmoid = nn.Sigmoid()

        self.bm_list1 = nn.ModuleList([AdaptiveModulator(self.hidden_dim) for _ in range(self.layer_num)])
        self.sm_list1 = nn.ModuleList([nn.Linear(self.embed_dims[-1] if i == 0 else self.hidden_dim,
                                                 self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim)
                                       for i in range(self.layer_num)])
        self.sigmoid1 = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        device = x.device
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if self.model == 'SwinJSCC_w/o_SAandRA':
            return self.head(x)

        # SNR-adaptive (SA)
        if self.model == 'SwinJSCC_w/_SA':
            snr_batch = torch.tensor(self.snr, dtype=torch.float, device=device).unsqueeze(0).expand(B, -1)
            temp = self.sm_list[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand_as(temp)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            return self.head(x)

        # Rate-adaptive (RA)
        if self.model == 'SwinJSCC_w/_RA':
            rate_batch = torch.tensor(self.rate, dtype=torch.float, device=device).unsqueeze(0).expand(B, -1)
            temp = self.sm_list[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list[i](temp)
                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand_as(temp)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

            mask_scores = torch.sum(mod_val, dim=1)                # B, L
            _, indices = mask_scores.sort(dim=1, descending=True)
            c_indices = indices[:, :self.rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, self.rate)
            c_indices = c_indices + add
            mask = torch.zeros_like(mask_scores).view(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.view(B, x.size(2)).unsqueeze(1).expand(-1, x.size(1), -1)  # B, L, C (broadcast later)
            x = x * mask
            return x, mask

        # SA + RA
        if self.model == 'SwinJSCC_w/_SAandRA':
            snr_batch = torch.tensor(self.snr, dtype=torch.float, device=device).unsqueeze(0).expand(B, -1)
            rate_batch = torch.tensor(self.rate, dtype=torch.float, device=device).unsqueeze(0).expand(B, -1)

            temp = self.sm_list1[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list1[i](temp)
                bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand_as(temp)
                temp = temp * bm
            mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
            x = x * mod_val1

            temp = self.sm_list[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list[i](temp)
                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand_as(temp)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

            mask_scores = torch.sum(mod_val, dim=1)
            _, indices = mask_scores.sort(dim=1, descending=True)
            c_indices = indices[:, :self.rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, self.rate)
            c_indices = c_indices + add
            mask = torch.zeros_like(mask_scores).view(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.view(B, x.size(2)).unsqueeze(1).expand(-1, x.size(1), -1)
            x = x * mask
            return x, mask

        raise ValueError(f"Unknown model mode: {self.model}")

    def flops(self) -> int:
        fl = self.patch_embed.flops()
        for layer in self.layers:
            fl += layer.flops()
        fl += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return fl

    def update_resolution(self, H: int, W: int) -> None:
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)), W // (2 ** (i_layer + 1)))


class DecoderBasicLayer(nn.Module):
    """One decoder stage: Swin blocks at current res, then optional upsample."""
    def __init__(self, dim: int, out_dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float=4., qkv_bias: bool=True, qk_scale: Optional[float]=None,
                 norm_layer=nn.LayerNorm, upsample: Optional[nn.Module]=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer) \
            if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def update_resolution(self, H: int, W: int) -> None:
        self.input_resolution = (H, W)
        for blk in self.blocks:
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class SwinJSCCDecoder(nn.Module):
    def __init__(
            self, 
            img_size: Tuple[int, int], 
            embed_dims: Iterable[int], 
            depths: Iterable[int],
            num_heads: Iterable[int], 
            C: Optional[int], 
            window_size: int=4, 
            mlp_ratio: float=4.,
            qkv_bias: bool=True, 
            qk_scale: Optional[float]=None, 
            norm_layer=nn.LayerNorm,
            ape: bool=False, 
            patch_norm: bool=True, 
            bottleneck_dim: int=16, 
            model: str='SwinJSCC_w/o_SAandRA',
            snr: float=10.0
        ):
        super().__init__()
        embed_dims = list(embed_dims)
        depths = list(depths)
        num_heads = list(num_heads)

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        num_patches = (self.H // 4) * (self.W // 4)
        self.model = model
        self.snr = snr

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DecoderBasicLayer(
                dim=int(embed_dims[i_layer]),
                out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                  self.patches_resolution[1] * (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, upsample=PatchReverseMerging
            )
            self.layers.append(layer)

        self.head = nn.Linear(C, embed_dims[0]) if C is not None else nn.Identity()

        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = 7
        self.bm_list = nn.ModuleList([AdaptiveModulator(self.hidden_dim) for _ in range(self.layer_num)])
        self.sm_list = nn.ModuleList([nn.Linear(self.embed_dims[0] if i == 0 else self.hidden_dim,
                                                self.embed_dims[0] if i == self.layer_num - 1 else self.hidden_dim)
                                      for i in range(self.layer_num)])
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model == 'SwinJSCC_w/o_SAandRA':
            x = self.head(x)
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.view(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif self.model == 'SwinJSCC_w/_SA':
            B, L, C = x.size()
            x = self.head(x)
            snr_batch = torch.tensor(self.snr, dtype=torch.float, device=x.device).unsqueeze(0).expand(B, -1)
            temp = self.sm_list[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(B, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.view(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif self.model == 'SwinJSCC_w/_RA':
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.view(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif self.model == 'SwinJSCC_w/_SAandRA':
            B, L, C = x.size()
            snr_batch = torch.tensor(self.snr, dtype=torch.float, device=x.device).unsqueeze(0).expand(B, -1)
            temp = self.sm_list[0](x.detach())
            for i in range(1, self.layer_num):
                temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(B, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.view(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        raise ValueError(f"Unknown model mode: {self.model}")

    def flops(self) -> int:
        return sum(layer.flops() for layer in self.layers)

    def update_resolution(self, H: int, W: int) -> None:
        self.H = H * (2 ** len(self.layers))
        self.W = W * (2 ** len(self.layers))
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer), W * (2 ** i_layer))