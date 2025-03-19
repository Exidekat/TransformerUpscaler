#!/usr/bin/env python
"""
EfficientTransformer/model.py

Revised Transformer-based model for image upscaling using a hierarchical transformer
architecture with an efficient (Linformer-style) attention mechanism.

Input: Low resolution image (B, 3, H, W) with arbitrary resolution (e.g., 720×1280)
Output: Upscaled image (B, 3, target_H, target_W) (e.g., 1080×1920)

Architecture:
  1. Encoder: Two convolutional layers (with ReLU) extract low-level features.
  2. Downsampling: A convolution with stride=2 reduces spatial dimensions.
  3. Hierarchical Transformer Stage: The downsampled feature map is padded (if needed)
     so that its spatial dimensions are divisible by window_size, then partitioned into non-overlapping
     windows and processed by transformer blocks using efficient (Linformer-style) attention.
  4. Decoder: A simple CNN refines the processed features to predict a residual image.
  5. Global Residual: The input is bicubically upsampled to the target resolution and added to the predicted residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### Helper functions for window partitioning
def window_partition(x, window_size):
    """
    Partitions input tensor x into non-overlapping windows.
    Input:
      x: (B, H, W, C)
      window_size: int
    Returns:
      windows: (B, num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reconstruct the original tensor from windows.
    Input:
      windows: (B, num_windows, window_size*window_size, C)
      window_size: int
      H, W: height and width of padded feature map
    Returns:
      x: (B, H, W, C)
    """
    B = windows.shape[0]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

### Efficient Window Attention Module (Linformer-style)
class EfficientWindowAttention(nn.Module):
    """
    Efficient window-based multi-head self attention with a Linformer-style projection.
    Projects keys and values from length N to a lower fixed dimension (proj_dim) to reduce
    computational and memory costs.
    """
    def __init__(self, dim, window_size, num_heads, proj_dim=16, dropout=0.0):
        super(EfficientWindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # assume square window: window_size x window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_dim = proj_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Linformer projection layers for keys and values.
        self.proj_k = nn.Linear(window_size * window_size, proj_dim, bias=False)
        self.proj_v = nn.Linear(window_size * window_size, proj_dim, bias=False)

        # Relative positional bias table.
        num_relative_distance = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
        )
        # Compute relative position index.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # (2, window_size*window_size)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        # x: (B*num_windows, N, C) with N = window_size*window_size.
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)
        q = q * self.scale

        # Project keys and values: transpose k and v to (B_, num_heads, head_dim, N)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # Apply Linformer projection: reduce sequence length from N to proj_dim.
        k = self.proj_k(k)  # (B_, num_heads, head_dim, proj_dim)
        v = self.proj_v(v)  # (B_, num_heads, head_dim, proj_dim)
        # Transpose back: now k and v become (B_, num_heads, proj_dim, head_dim)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # Compute attention: q: (B_, num_heads, N, head_dim), k: (B_, num_heads, proj_dim, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B_, num_heads, N, proj_dim)

        # Compute relative positional bias.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        # Permute to (1, num_heads, N, N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        # Adjust the bias by slicing the last dimension to match proj_dim.
        relative_position_bias = relative_position_bias[:, :, :, :attn.shape[-1]]  # (1, num_heads, N, proj_dim)

        attn = attn + relative_position_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)  # (B_, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

### Efficient Window Transformer Block
class EfficientWindowTransformerBlock(nn.Module):
    """
    Transformer block operating on local windows using EfficientWindowAttention.
    """
    def __init__(self, dim, window_size, num_heads, proj_dim=16, mlp_ratio=4.0, dropout=0.0):
        super(EfficientWindowTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientWindowAttention(dim, window_size, num_heads, proj_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: (B, num_windows, N, C)
        B, num_windows, N, C = x.shape
        x = x.view(B * num_windows, N, C)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, num_windows, N, C)
        return x

### Efficient Transformer Model
class TransformerModel(nn.Module):
    def __init__(
            self,
            in_channels=3,
            base_channels=64,
            num_transformer_blocks=4,
            num_heads=4,
            proj_dim=16,
            mlp_ratio=4.0,
            dropout=0.0,
            window_size=8
    ):
        super(TransformerModel, self).__init__()
        # Encoder: two conv layers with ReLU.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Downsample: reduce resolution by a factor of 2.
        self.downsample = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.window_size = window_size
        # Hierarchical Transformer stage: process feature map windows using efficient transformer blocks.
        self.blocks = nn.ModuleList([
            EfficientWindowTransformerBlock(dim=base_channels, window_size=window_size, num_heads=num_heads,
                                            proj_dim=proj_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        # Decoder: a simple CNN to predict the residual.
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.output_activation = nn.Sigmoid()

    def transformer_stage(self, x):
        """
        Apply the hierarchical transformer stage to x.
        x: (B, C, H, W) from the downsampled encoder output.
        """
        B, C, H, W = x.shape
        # Pad H and W so they are multiples of window_size.
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size
        pad_right = (self.window_size - W % self.window_size) % self.window_size
        if pad_bottom != 0 or pad_right != 0:
            x = F.pad(x, (0, pad_right, 0, pad_bottom))
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)
        windows = window_partition(x, self.window_size)  # (B, num_windows, window_size*window_size, C)
        for blk in self.blocks:
            windows = blk(windows)
        x = window_reverse(windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)
        if pad_bottom != 0 or pad_right != 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # back to (B, C, H, W)
        return x

    def forward(self, x, res_out=(1080, 1920)):
        """
        x: (B, 3, H, W) with arbitrary resolution.
        res_out: desired output resolution (target_H, target_W)
        """
        upscaled_input = F.interpolate(x, size=res_out, mode='bicubic', align_corners=False)
        features = self.encoder(x)  # (B, base_channels, H, W)
        features_down = self.downsample(features)  # (B, base_channels, H/2, W/2)
        features_transformed = self.transformer_stage(features_down)  # (B, base_channels, H/2, W/2)
        residual = self.decoder_conv(features_transformed)  # (B, 3, H/2, W/2)
        residual_up = F.interpolate(residual, size=res_out, mode='bicubic', align_corners=False)
        out = upscaled_input + residual_up
        out = self.output_activation(out)
        return out

# Quick test when running this file directly.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)
