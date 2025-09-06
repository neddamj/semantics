import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, to_2tuple

"""
    WITT: A Wireless Image Transmission Transformer for Semantic Communications.
    Paper: https://arxiv.org/pdf/2211.00937
"""

class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def window_partition(x, window_size):
    """Partitions a feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverses the window partitioning."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Define and register relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        x = (self.softmax(attn) @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

        # Pre-calculate attention mask for shifted window attention
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)
        
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = x.view(B, L, C)
        
        return shortcut + x + self.mlp(self.norm2(x))

class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling."""
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1) # B, H/2, W/2, 4*C
        x = self.norm(x.view(B, -1, 4 * C))
        return self.reduction(x)

class PatchReverseMerging(nn.Module):
    """Patch Reverse Merging Layer for upsampling using pixel shuffle."""
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expansion = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expansion(self.norm(x))
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, 2)
        return x.flatten(2).transpose(1, 2)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)

class FeatureModulator(nn.Module):
    """Modulates features using a gating mechanism controlled by SNR."""
    def __init__(self, dim, mod_layers=7):
        super().__init__()
        self.mod_layers = mod_layers
        hidden_dim = int(dim * 1.5)

        # A list of MLPs to generate modulation signals from SNR
        self.snr_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            for _ in range(mod_layers)
        ])
        
        # A list of Linear layers to process the feature tensor
        self.feature_mlps = nn.ModuleList([nn.Linear(dim, hidden_dim)])
        for _ in range(mod_layers - 1):
            self.feature_mlps.append(nn.Linear(hidden_dim, hidden_dim))
        self.feature_mlps.append(nn.Linear(hidden_dim, dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr):
        B, L, _ = x.shape
        snr_tensor = torch.tensor(snr, dtype=torch.float, device=x.device).expand(B, 1)
        
        temp = x
        for i in range(self.mod_layers):
            features_in = temp.detach() if i == 0 else temp
            temp = self.feature_mlps[i](features_in)
            mod_signal = self.snr_mlps[i](snr_tensor).unsqueeze(1).expand(-1, L, -1)
            temp = temp * mod_signal
        
        gate = self.sigmoid(self.feature_mlps[-1](temp))
        return x * gate

class EncoderStage(nn.Module):
    """A single stage of the Swin Transformer Encoder."""
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., downsample=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.downsample = PatchMerging(input_resolution, dim, out_dim, norm_layer) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x)

class DecoderStage(nn.Module):
    """A single stage of the Swin Transformer Decoder."""
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., upsample=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.upsample = PatchReverseMerging(input_resolution, dim, out_dim, norm_layer) if upsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.upsample(x)

class WITTEncoder(nn.Module):
    """Swin Transformer Encoder."""
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        embed_dims=[64, 128],
        depths=[2, 2],
        num_heads=[2, 4],
        C_out=32,
        window_size=4,
        use_modulation=True,
        snr=10.0,
        in_chans=3
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        res = self.patch_embed.patches_resolution
        self.snr = snr
        
        self.stages = nn.ModuleList()
        for i, (dim, out_dim, depth, heads) in enumerate(zip(embed_dims, embed_dims[1:] + [embed_dims[-1]], depths, num_heads)):
            self.stages.append(
                EncoderStage(
                    dim=dim,
                    out_dim=out_dim,
                    input_resolution=(res[0] // (2 ** i), res[1] // (2 ** i)),
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    downsample=(i < len(embed_dims) - 1)
                )
            )

        final_dim = embed_dims[-1]
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, C_out)
        self.modulator = FeatureModulator(final_dim) if use_modulation else None

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        if self.modulator:
            x = self.modulator(x, self.snr)
        return self.head(x)

class WITTDecoder(nn.Module):
    """Swin Transformer Decoder."""
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        embed_dims=[128, 64],
        depths=[2, 2],
        num_heads=[4, 2],
        C_in=32,
        window_size=4,
        use_modulation=True,
        snr=10.0,
        out_chans=3
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.H, self.W = img_size, img_size
        # tokens resolution at decoder input
        self.res = (
            img_size // (patch_size * (2 ** (len(depths) - 1))),
            img_size // (patch_size * (2 ** (len(depths) - 1))),
        )
        self.snr = snr
        self.out_chans = out_chans  # <--- store it

        initial_dim = embed_dims[0]
        self.head = nn.Linear(C_in, initial_dim)
        self.modulator = FeatureModulator(initial_dim) if use_modulation else None

        self.stages = nn.ModuleList()
        # IMPORTANT: last stage should output 'out_chans' (NOT hard-coded 3)
        for i, (dim, out_dim, depth, heads) in enumerate(
            zip(embed_dims, embed_dims[1:] + [out_chans], depths, num_heads)  # <--- here
        ):
            self.stages.append(
                DecoderStage(
                    dim=dim,
                    out_dim=out_dim,
                    input_resolution=(self.res[0] * (2 ** i), self.res[1] * (2 ** i)),
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    upsample=True,
                )
            )

    def forward(self, x):
        x = self.head(x)
        if self.modulator:
            x = self.modulator(x, self.snr)
        for stage in self.stages:
            x = stage(x)
        # Use out_chans and reshape (safer than view for non-contiguous tensors)
        return x.reshape(-1, self.out_chans, self.H, self.W).sigmoid()

class WITTransformer(nn.Module):
    """WIT Transformer: Combines WITT Encoder and Decoder."""
    def __init__(self, encoder_cfg, decoder_cfg):
        super().__init__()
        self.encoder = WITTEncoder(**encoder_cfg)
        self.decoder = WITTDecoder(**decoder_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon