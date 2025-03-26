#!/usr/bin/env python
"""
TransformerModel.py

Revised Transformer-based model for image upscaling using relative positional encoding.

Architecture Overview:
  1. A shallow CNN encoder extracts features from the low-resolution input.
  2. A downsampling layer reduces spatial dimensions.
  3. A convolutional patch embedding converts the feature map into a grid of tokens.
  4. The token grid is partitioned into non-overlapping windows.
  5. A series of window transformer blocks (with relative positional encoding)
     process the tokens.
  6. The windows are merged back into a 2D token grid.
  7. A patch unembedding layer (transposed convolution) converts tokens back to a feature map.
  8. A CNN decoder refines the features and predicts a residual image.
  9. A global residual connection adds the predicted residual to a bicubically upscaled input.

This design is resolution-agnostic because the patch embedding and window operations
handle arbitrary input sizes, and the global residual is computed via bicubic upsampling.
A minor spatial mismatch (if dimensions are not divisible by the window size) is resolved via cropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition the input tensor into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Size of the square window.

    Returns:
        torch.Tensor: Windows of shape (B, num_windows, window_size*window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, window_size * window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse the window partition process to reconstruct the tensor.

    Args:
        windows (torch.Tensor): Tensor of shape (B, num_windows, window_size*window_size, C).
        window_size (int): Size of the square window.
        H (int): Height of the padded feature map.
        W (int): Width of the padded feature map.

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, H, W, C).
    """
    B = windows.shape[0]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative positional encoding.
    Inspired by the Swin Transformer.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Assume square window: window_size x window_size.
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Relative positional bias table.
        num_relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        # Compute relative position index.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # (2, window_size*window_size)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        # Shift to start from 0.
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size * self.window_size,
                                                             self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class WindowTransformerBlock(nn.Module):
    """
    Transformer block operating on local windows with relative positional encoding.
    Consists of a window attention layer followed by an MLP with residual connections.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(WindowTransformerBlock, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim).
        """
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class TransformerModel(nn.Module):
    """
    Transformer-based model for image upscaling using relative positional encoding.

    Architecture:
      1. CNN encoder extracts features from the low-resolution input.
      2. Downsampling reduces spatial dimensions.
      3. Patch embedding converts features to a grid of tokens.
      4. Tokens are partitioned into non-overlapping windows.
      5. Window transformer blocks process the tokens.
      6. Windows are merged back into a token grid.
      7. Patch unembedding reconstructs a feature map from tokens.
      8. CNN decoder refines features to predict a residual image.
      9. Global residual connection: The predicted residual is added to a bicubically upscaled input.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        transformer_dim: int = 128,
        num_window_blocks: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.01,
        window_size: int = 8,
    ):
        super(TransformerModel, self).__init__()
        # Encoder: Shallow CNN.
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling layer.
        self.downsample = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        # Patch embedding: converts feature map to tokens.
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=8, stride=8)

        # Window transformer blocks.
        self.window_size = window_size
        self.window_blocks = nn.ModuleList([
            WindowTransformerBlock(transformer_dim, window_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_window_blocks)
        ])

        # Patch unembedding: converts tokens back to a feature map.
        self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels, kernel_size=8, stride=8)

        # Decoder: CNN layers to predict the residual image.
        self.decoder_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.decoder_conv2 = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, res_out: Tuple[int, int] = (1080, 1920), upscale_factor: int = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Target output resolution (height, width).
            upscale_factor (int): Upscale factor (optional)

        Returns:
            torch.Tensor: Upscaled image of shape (B, 3, target_H, target_W).
        """
        # Compute target resolution.
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)

        # Global residual: bicubic upscale of input.
        upscaled_input = F.interpolate(x, size=res_out, mode='bicubic', align_corners=False)

        # Encoder.
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))

        # Downsample features.
        feat_down = self.downsample(feat)  # (B, base_channels, H_d, W_d)

        # Patch embedding.
        tokens = self.patch_embed(feat_down)  # (B, transformer_dim, H_t, W_t)
        B, C, H_t, W_t = tokens.shape
        # Rearrange tokens to (B, H_t, W_t, transformer_dim)
        tokens = tokens.permute(0, 2, 3, 1).contiguous()

        # Pad token grid if necessary for window partitioning.
        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right = (self.window_size - W_t % self.window_size) % self.window_size
        orig_H, orig_W = H_t, W_t
        if pad_bottom > 0 or pad_right > 0:
            tokens = tokens.permute(0, 3, 1, 2)  # (B, transformer_dim, H_t, W_t)
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom))
            tokens = tokens.permute(0, 2, 3, 1).contiguous()  # (B, H_pad, W_pad, transformer_dim)
            H_t, W_t = tokens.shape[1], tokens.shape[2]

        # Partition tokens into windows.
        tokens_windows = window_partition(tokens, self.window_size)  # (B, num_windows, window_size*window_size, transformer_dim)
        B, num_windows, N, D = tokens_windows.shape
        tokens_windows = tokens_windows.view(B * num_windows, N, D)

        # Process windows with transformer blocks.
        for block in self.window_blocks:
            tokens_windows = block(tokens_windows)

        # Merge windows back to token grid.
        tokens_windows = tokens_windows.view(B, num_windows, N, D)
        tokens = window_reverse(tokens_windows, self.window_size, H_t, W_t)  # (B, H_t, W_t, transformer_dim)

        # Remove padding if added.
        if pad_bottom > 0 or pad_right > 0:
            tokens = tokens[:, :orig_H, :orig_W, :]

        # Rearrange tokens back to (B, transformer_dim, H_t, W_t).
        tokens = tokens.permute(0, 3, 1, 2).contiguous()

        # Patch unembedding.
        feat_trans = self.patch_unembed(tokens)  # (B, base_channels, H_d_un, W_d_un)

        # Crop features for skip connection compatibility.
        min_h = min(feat_down.shape[2], feat_trans.shape[2])
        min_w = min(feat_down.shape[3], feat_trans.shape[3])
        feat_down = feat_down[:, :, :min_h, :min_w]
        feat_trans = feat_trans[:, :, :min_h, :min_w]
        combined_feat = feat_down + feat_trans

        # Decoder.
        dec = self.relu(self.decoder_conv1(combined_feat))
        residual = self.decoder_conv2(dec)

        # Upsample the predicted residual.
        residual_up = F.interpolate(residual, size=res_out, mode='bicubic', align_corners=False)

        # Final output.
        out = upscaled_input + residual_up
        return torch.clamp(out, 0.0, 1.0)

# Quick test.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)