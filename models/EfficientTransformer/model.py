#!/usr/bin/env python
"""
TransformerModel.py

Revised Transformer-based model for image upscaling using relative positional encoding.

This architecture incorporates:
  - A shallow CNN encoder that extracts features from the low-resolution input.
  - A downsampling layer that reduces spatial dimensions.
  - A convolutional patch embedding that converts the feature map into a grid of tokens.
  - Local window transformer blocks with relative positional encoding (inspired by Swin Transformer)
    that operate on non-overlapping windows.
  - A patch unembedding layer that reconstructs a feature map from the processed tokens.
  - A CNN decoder that refines the features and predicts a residual image.
  - A global residual connection: the predicted residual is added to a bicubically upscaled input.

This design is resolution-agnostic because the patch embedding and window operations handle arbitrary
input sizes, and the global residual is computed via bicubic upsampling. A minor spatial mismatch due
to non-divisible dimensions is resolved via cropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Partitions input tensor x into non-overlapping windows.

    Args:
      x: Tensor of shape (B, H, W, C)
      window_size: int, size of the window

    Returns:
      windows: Tensor of shape (B, num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reconstructs the tensor from windows.

    Args:
      windows: Tensor of shape (B, num_windows, window_size*window_size, C)
      window_size: int, size of the window used during partitioning
      H, W: int, height and width of the padded feature map

    Returns:
      x: Tensor of shape (B, H, W, C)
    """
    B = windows.shape[0]
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self-attention with relative positional encoding.
    Inspired by the Swin Transformer.
    """

    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # assume square window: window_size x window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Relative positional bias table.
        num_relative_positions = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        # Compute relative position index.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, window_size, window_size)
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
        # x: (B, N, dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size * self.window_size,
                                                             self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class WindowTransformerBlock(nn.Module):
    """
    Transformer block operating on local windows with relative positional encoding.
    """

    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(WindowTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, dim) where N = window_size*window_size
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
    Revised TransformerModel for image upscaling using relative positional encoding.

    Architecture overview:
      1. A shallow CNN encoder extracts features from the low-resolution input.
      2. A downsampling layer reduces spatial dimensions.
      3. A patch embedding layer (convolution) converts the feature map into a grid of tokens.
      4. The token grid is partitioned into non-overlapping windows.
      5. A series of window transformer blocks (with relative positional encoding) process the tokens.
      6. The windows are merged back into a 2D token grid.
      7. A patch unembedding layer (transposed convolution) converts tokens back to a feature map.
      8. A CNN decoder refines the features and predicts a residual image.
      9. The predicted residual is added to a bicubically upscaled input (global residual) to produce the final output.
    """

    def __init__(
            self,
            in_channels=3,
            base_channels=64,
            transformer_dim=128,
            num_window_blocks=8,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            window_size=8
    ):
        super(TransformerModel, self).__init__()
        # Encoder: Shallow CNN.
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling: e.g., from (H, W) to roughly (H/2, W/2).
        self.downsample = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        # Patch Embedding: converts feature map to tokens.
        # For example, with kernel_size=8 and stride=8, a feature map of size (H_d, W_d) becomes a token grid.
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=8, stride=8)

        # Window Transformer Blocks.
        self.window_size = window_size
        self.window_blocks = nn.ModuleList([
            WindowTransformerBlock(transformer_dim, window_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_window_blocks)
        ])

        # Patch Unembedding: converts tokens back to feature map.
        self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels, kernel_size=8, stride=8)

        # Decoder: CNN decoder to predict the residual image.
        self.decoder_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.decoder_conv2 = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, res_out=(1080, 1920)):
        """
        Forward pass:
          x: Input image tensor of shape (B, 3, H, W) with arbitrary resolution.
          res_out: Desired output resolution (target_H, target_W).
        """
        # Global residual: bicubic upscaling of input.
        upscaled_input = F.interpolate(x, size=res_out, mode='bicubic', align_corners=False)

        # Encoder.
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))

        # Downsample.
        feat_down = self.downsample(feat)  # (B, base_channels, H_d, W_d)

        # Patch Embedding.
        tokens = self.patch_embed(feat_down)  # (B, transformer_dim, H_t, W_t)
        B, C, H_t, W_t = tokens.shape
        # Rearrange tokens to (B, H_t, W_t, transformer_dim)
        tokens = tokens.permute(0, 2, 3, 1).contiguous()

        # Pad token grid if necessary so that H_t and W_t are divisible by window_size.
        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right = (self.window_size - W_t % self.window_size) % self.window_size
        orig_H, orig_W = H_t, W_t
        if pad_bottom > 0 or pad_right > 0:
            tokens = tokens.permute(0, 3, 1, 2)  # (B, transformer_dim, H_t, W_t)
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom))
            tokens = tokens.permute(0, 2, 3, 1)  # (B, H_pad, W_pad, transformer_dim)
            H_t, W_t = tokens.shape[1], tokens.shape[2]

        # Partition tokens into windows.
        tokens_windows = window_partition(tokens,
                                          self.window_size)  # (B, num_windows, window_size*window_size, transformer_dim)
        B, num_windows, N, D = tokens_windows.shape
        tokens_windows = tokens_windows.reshape(B * num_windows, N, D)

        # Process each window with window transformer blocks.
        for block in self.window_blocks:
            tokens_windows = block(tokens_windows)

        # Merge windows back to token grid.
        tokens_windows = tokens_windows.reshape(B, num_windows, N, D)
        tokens = window_reverse(tokens_windows, self.window_size, H_t, W_t)  # (B, H_t, W_t, transformer_dim)

        # Remove padding if added.
        if pad_bottom > 0 or pad_right > 0:
            tokens = tokens[:, :orig_H, :orig_W, :]

        # Rearrange tokens to (B, transformer_dim, H_t, W_t)
        tokens = tokens.permute(0, 3, 1, 2).contiguous()

        # Patch Unembedding.
        feat_trans = self.patch_unembed(tokens)  # (B, base_channels, H_d_un, W_d_un)

        # To ensure skip connection compatibility, crop feat_down and feat_trans to the same size.
        min_h = min(feat_down.shape[2], feat_trans.shape[2])
        min_w = min(feat_down.shape[3], feat_trans.shape[3])
        feat_down = feat_down[:, :, :min_h, :min_w]
        feat_trans = feat_trans[:, :, :min_h, :min_w]
        combined_feat = feat_down + feat_trans

        # Decoder.
        dec = self.relu(self.decoder_conv1(combined_feat))
        residual = self.decoder_conv2(dec)

        # Upsample residual to desired output resolution.
        residual_up = F.interpolate(residual, size=res_out, mode='bicubic', align_corners=False)

        # Final output.
        out = upscaled_input + residual_up
        out = torch.clamp(out, 0.0, 1.0)
        return out


# Quick test when running this file directly.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)
