#!/usr/bin/env python


from email.mime import base
import math
from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple
import time


# -----------------------------
# Window partition/reverse
# -----------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, window_size * window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = windows.shape[0]
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


# -----------------------------
# Patch modules
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # Patch embed: downsample spatial dims by factor 8 to reduce token count
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=8)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.norm(x)
        return x

class PatchUnembedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Patch unembed: inverse of patch embedding
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=8, stride=8)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.conv(x)

class TransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, H, W, C) → conv expects (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # back to (B, H, W, C)
        return x

# -----------------------------
# Basic blocks (Conv, Upsampler, CA, ResBlock)
# -----------------------------
def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias, groups=groups)

class Upsampler(nn.Module):
    def __init__(self, conv, n_feats, valid_scales=(2, 3, 4, 6), bn=False, act=False, bias=True):
        super().__init__()
        self.upsamplers = nn.ModuleDict()
        for scale in valid_scales:
            blocks = []
            if (scale & (scale - 1)) == 0:
                steps = int(math.log2(scale))
                for _ in range(steps):
                    blocks.append(conv(n_feats, 4 * n_feats, 3, bias))
                    blocks.append(nn.PixelShuffle(2))
                    if bn:
                        blocks.append(nn.BatchNorm2d(n_feats))
                    if act == 'relu':
                        blocks.append(nn.ReLU(True))
                    elif act == 'prelu':
                        blocks.append(nn.PReLU(n_feats))
            elif scale == 3:
                blocks.append(conv(n_feats, 9 * n_feats, 3, bias))
                blocks.append(nn.PixelShuffle(3))
                if bn:
                    blocks.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    blocks.append(nn.ReLU(True))
                elif act == 'prelu':
                    blocks.append(nn.PReLU(n_feats))
            elif scale == 6:
                blocks.append(conv(n_feats, 36 * n_feats, 3, bias))
                blocks.append(nn.PixelShuffle(6))
                if bn:
                    blocks.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    blocks.append(nn.ReLU(True))
                elif act == 'prelu':
                    blocks.append(nn.PReLU(n_feats))
            else:
                raise NotImplementedError(f"Scale={scale} not supported")
            self.upsamplers[str(scale)] = nn.Sequential(*blocks)

    def forward(self, x, scale):
        scale_str = str(scale)
        if scale_str not in self.upsamplers:
            raise ValueError(f"Requested scale={scale} not built.")
        return self.upsamplers[scale_str](x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  
        y = self.conv_du(y)   
        return x * y         

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, act=nn.ReLU(True), use_ca=True, reduction=8):
        super().__init__()
        modules = [
            default_conv(n_feats, n_feats, kernel_size, bias=bias),
            act,
            default_conv(n_feats, n_feats, kernel_size, bias=bias)
        ]
        if use_ca:
            modules.append(CALayer(n_feats, reduction))
        self.body = nn.Sequential(*modules)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# -----------------------------
# Window/Shifted Attention
# -----------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Relative positional bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))  # (2, window_size, window_size)
        coords_flatten = coords.flatten(1)  # (2, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class ShiftedWindowTransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 mlp_ratio=4.0, dropout=0.1, shift_size=0):
        super().__init__()
        self.window_size = window_size
        # Even if shift_size is 0, by default we do half-window shift:
        self.shift_size = shift_size 
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    @staticmethod    
    def get_attn_mask(H, W, window_size, shift_size, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            for w in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # (B*nW, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)

        # Pad for window partition
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size
        pad_right = (self.window_size - W % self.window_size) % self.window_size
        H_t, W_t = H + pad_bottom, W + pad_right

        if pad_bottom > 0 or pad_right > 0:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_right, 0, pad_bottom))
            x = x.permute(0, 2, 3, 1).contiguous()

        # Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        windows = window_partition(shifted_x, self.window_size)
        B_win, num_windows, N, _ = windows.shape
        windows = windows.view(B_win * num_windows, N, C)

        # Attention
        attn_mask = self.get_attn_mask(H_t, W_t, self.window_size, self.shift_size, x.device) if self.shift_size > 0 else None
        attn_windows = self.attn(windows, mask=attn_mask)
        attn_windows = attn_windows.view(B_win, num_windows, N, C)

        # Reverse windows
        shifted_x = window_reverse(attn_windows, self.window_size, H_t, W_t)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # Unpad
        if pad_bottom > 0 or pad_right > 0:
            x = x[:, :H, :W, :].contiguous()

        # MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.view(B, -1, C)).view(B, H, W, C))
        return x

# -----------------------------
# Coarse Global Attention
# -----------------------------
class CoarseGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, pool_size=4, qkv_bias=True, dropout=0.):
        super().__init__()
        self.pool_size = pool_size
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Learnable downscaling conv layer
        self.downscale = nn.Conv2d(
            dim, dim, kernel_size=pool_size, stride=pool_size, padding=0, groups=dim  # depthwise
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x

        # (B, H, W, C) -> (B, C, H, W)
        x_ch = x.permute(0, 3, 1, 2).contiguous()

        # Learnable downscaling
        pooled = self.downscale(x_ch)  # (B, C, H_p, W_p)
        H_p, W_p = pooled.shape[2], pooled.shape[3]
        N_p = H_p * W_p

        # (B, C, H_p, W_p) -> (B, N_p, C)
        pooled = pooled.flatten(2).transpose(1, 2)

        # QKV projection
        qkv = self.qkv(pooled).reshape(B, N_p, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N_p, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Output projection
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N_p, C)
        attn_out = self.proj_drop(self.proj(attn_out))

        # (B, N_p, C) -> (B, C, H_p, W_p)
        attn_out = attn_out.transpose(1, 2).view(B, C, H_p, W_p)

        # Upsample back to (H, W)
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        out = attn_out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        return shortcut + out

# -----------------------------
# HybridAttentionBlock + Residual Container
# -----------------------------
class HybridAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: float = 1.5,
        dropout: float = 0.1,
        channel_reduction: int = 16,
        shift_size: int = 0,
    ):
        super().__init__()
        self.ca = CALayer(channel=dim, reduction=channel_reduction)
        self.shift_attn = ShiftedWindowTransformerBlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            shift_size=shift_size
        )
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # LN
        x_norm = self.norm1(x.view(B, -1, C)).view(B, H, W, C)
        shortcut = x_norm

        # Channel Attention
        ca_x = x_norm.permute(0, 3, 1, 2)
        ca_x = self.ca(ca_x)
        ca_x = ca_x.permute(0, 2, 3, 1)

        # Shifted Window
        sw_x = self.shift_attn(x_norm)

        fused = shortcut + ca_x + sw_x

        # MLP
        fused_norm = self.norm2(fused.view(B, -1, C)).view(B, H, W, C)
        mlp_out = self.mlp(fused_norm.view(B, -1, C)).view(B, H, W, C)
        out = fused + mlp_out
        return out
    
class HAI(nn.Module):
    def __init__(self,
                 transformer_block: nn.Module,
                 dim: int,
                 ):
        super().__init__()
        
        self.block = transformer_block
        self.alpha = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor):
        out = self.block(x)
        alpha = self.alpha.view(1, 1, 1, -1)
        return out + alpha * x
        

class ResidualHABBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 window_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 channel_reduction: int = 16,
                 pool_size: int = 4,
                 blocks_per_group = 3
                 ):
        super().__init__()

        shift = window_size // 2
        blocks = []
        for i in range(blocks_per_group):
            shift_size = 0 if i % 2 == 0 else shift
            blocks.append(
                HAI(
                    HybridAttentionBlock(
                        dim=dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        channel_reduction=channel_reduction,
                        shift_size=shift_size
                    ),
                    dim=dim
                )
            )
            blocks.append(
                HAI(
                    CoarseGlobalAttention(
                        dim=dim,
                        num_heads=num_heads,
                        pool_size=pool_size,
                        qkv_bias=True,
                        dropout=dropout
                    ),
                    dim=dim
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )

        self.residual_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x

        x = self.blocks(x)  # still (B, H, W, C)

        # We need to permute for residual_conv
        x_perm = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x_perm = self.residual_conv(x_perm)
        x = x_perm.permute(0, 2, 3, 1).contiguous()  # back to (B, H, W, C)

        x = x + shortcut
        return x


# -----------------------------
# Full Model
# -----------------------------
class TransformerModel(nn.Module):
    """
    The main transformer-based upscaling model.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        transformer_dim: int = 60,
        num_groups: int = 6,
        blocks_per_group: int = 3,
        num_heads: int = 2,
        mlp_ratio: float = 2,
        dropout: float = 0.1,
        window_size: int = 8,
        coarse_attention_pool_size: int = 4
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # Shallow CNN encoder
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        # Learnable upscaling
        self.up_residual = Upsampler(conv=default_conv, n_feats=base_channels)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=base_channels,
            embed_dim=transformer_dim,
            norm_layer=nn.LayerNorm
        )

        self.body = nn.Sequential(
            *[ResidualHABBlock(dim=transformer_dim, window_size=window_size, num_heads=num_heads,
                               mlp_ratio=mlp_ratio, dropout=dropout, channel_reduction=coarse_attention_pool_size,
                               blocks_per_group=blocks_per_group) for _ in range(num_groups)]
        )
        self.deep_conv = TransformerConv(transformer_dim, transformer_dim)

        # Patch unembedding (final upsample from H/16 → H)
        self.patch_unembed = PatchUnembedding(
            in_channels=transformer_dim,
            out_channels=base_channels
        )

        # Residual branch for original input
        self.up1 = Upsampler(conv=default_conv, n_feats=in_channels)  # upsample the original x if you want
        self.up1_conv = default_conv(in_channels, base_channels, 3, bias=True)

        # Final upscaler (from base_channels → final resolution). 
        self.upsampler = Upsampler(
            conv=default_conv,
            n_feats=base_channels,
            valid_scales=(2, 3, 4, 6),
            act='relu'
        )

        self.to_rgb = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor,
                res_out: Tuple[int, int] = (1080, 1920),
                upscale_factor: int = None,
                require_ratio: bool = True) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        """
        # 1) Determine scale
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)
        else:
            upscale_factor = math.ceil(
                max(res_out[0] / x.shape[2], res_out[1] / x.shape[3])
            )
        
        # residual branch for original input
         
        B, C, H, W = x.shape 
        H_new = ((H + 8 - 1) // 8) * 8
        W_new = ((W + 8 - 1) // 8) * 8

        pad_h = H_new - H
        pad_w = W_new - W
    
        if pad_h > 0 or pad_w > 0:
            # [left, right, top, bottom]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 2) Shallow feature
        feat = self.relu(self.conv1(x))  # (B, base_channels, H, W)
        
        feat_residual = feat

        # 3) Patch embed
        tkns = self.patch_embed(feat)    # (B, H/4, W/4, transformer_dim)

        # 4) Stage 1
        tkns = self.body(tkns)
        tkns = self.deep_conv(tkns)

        out_feats = self.patch_unembed(tkns)
        
        
        out_feats = out_feats + feat_residual
        # out_feats = out_feats[:, :, :H, :W]

        # 8) Upscale to target
        out = self.upsampler(out_feats, scale=upscale_factor)  # (B, base_channels, H*, W*)

        # 9) Convert to RGB
        out = self.to_rgb(out)  # (B, 3, H*, W*)
        out = out[:, :, :H * upscale_factor, :W * upscale_factor]  # remove padding

        # 10) Optionally enforce aspect ratio
        if require_ratio and (out.shape[2], out.shape[3]) != res_out:
            hr_squash = transforms.Resize(res_out)
            out = hr_squash(out)

        return torch.clamp(out, 0.0, 1.0)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 100, 100).to(device)
    start = time.time()
    output = model(dummy_input, upscale_factor=6)
    end = time.time()
    print("Output shape:", output.shape)
    print("Time taken:", round(end - start, 4), 's')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
