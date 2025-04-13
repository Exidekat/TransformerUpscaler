#!/usr/bin/env python
"""
TransformerModel.py

This model performs image upscaling using a transformer architecture with relative positional encoding.
It extracts features using a CNN encoder, embeds them into a token grid via a convolution (patch embedding),
processes windows of tokens with transformer blocks (including window-based and shifted-window attention),
reconstructs a feature map via patch unembedding, and then upsamples and refines the result via a CNN decoder.
A global residual connection is applied using upscaled input and predicted residual.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple, Literal
import time

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Splits the input tensor into non-overlapping square windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): The dimension of the square window.

    Returns:
        torch.Tensor: Windows of shape (B, num_windows, window_size*window_size, C).
    """
    B, H, W, C = x.shape
    # Reshape into a grid of windows and then permute to flatten windows
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, window_size * window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reconstructs the original tensor from its window partitions.

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

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    """
    Returns a 2D convolution with weight normalization.
    """
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias, groups=groups)

class BasicConv(nn.Module):
    """
    A basic convolutional (or transposed convolution) block that may include batch norm,
    activation, and an optional upsampling operation.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1,
                 groups=1, relu=True, bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            # Use transposed convolution for upsampling if fan flag is True
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU(inplace=True) if relu else None
            self.up_size = up_size
            # If up_size is provided, include an upsampling layer
            self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class Upsampler(nn.Module):
    """
    Upsampler using sub-pixel convolution (PixelShuffle) for a set of fixed scale factors.
    It builds a dictionary of upsampling modules keyed by the scale factor.
    """
    def __init__(self, conv, n_feats, valid_scales=(2, 3, 4, 6), bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        self.upsamplers = nn.ModuleDict()
        for scale in valid_scales:
            blocks = []
            if (scale & (scale - 1)) == 0:
                # For scales that are powers of two
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
                # For scale factor 3
                blocks.append(conv(n_feats, 9 * n_feats, 3, bias))
                blocks.append(nn.PixelShuffle(3))
                if bn:
                    blocks.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    blocks.append(nn.ReLU(True))
                elif act == 'prelu':
                    blocks.append(nn.PReLU(n_feats))
            elif scale == 6:
                # For scale factor 6
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
            raise ValueError(f"Requested scale={scale} was not built.")
        return self.upsamplers[scale_str](x)

class CALayer(nn.Module):
    """
    Channel Attention layer from RCAN. It applies global average pooling followed by a bottleneck (1x1 convolutions)
    and a sigmoid to reweight the channels.
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)    # (B, C, 1, 1)
        y = self.conv_du(y)     # (B, C, 1, 1)
        return x * y            # Scale input with attention weights

class ResBlock(nn.Module):
    """
    Residual block with two convolutions, an optional activation, and an optional channel attention module.
    The residual is added back to the output.
    """
    def __init__(self, n_feats, kernel_size=3, bias=True, act=nn.ReLU(True), use_ca=True, reduction=8):
        super(ResBlock, self).__init__()
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
        res += x   # Add residual
        return res

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative positional encoding.
    It projects inputs down to a reduced dimension for efficiency, then computes attention,
    and finally projects the result back to the original dimension.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, reduction_ratio: int = 2, dropout: float = 0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        # Reduced dimension must evenly divide by num_heads
        self.reduced_dim = dim // reduction_ratio
        assert self.reduced_dim * reduction_ratio == dim, f"dim ({dim}) must be divisible by reduction_ratio ({reduction_ratio})"
        assert self.reduced_dim % num_heads == 0, f"reduced_dim ({self.reduced_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.reduced_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers: down-project, then QKV, then up-project.
        self.proj_down = nn.Linear(dim, self.reduced_dim, bias=True)
        self.qkv = nn.Linear(self.reduced_dim, self.reduced_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_up = nn.Linear(self.reduced_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

        # Build the relative positional bias table and index.
        num_relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape (B, N, C), where N is the number of tokens in a window.
        Returns:
            Tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        assert C == self.dim, "Input dimension mismatch"

        # Project input down
        x_reduced = self.proj_down(x)  # (B, N, reduced_dim)

        # Compute QKV from reduced representation
        qkv = self.qkv(x_reduced).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        q = q * self.scale
        # Compute attention scores and add relative positional bias
        attn = torch.matmul(q, k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute weighted sum of values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.reduced_dim)
        out = self.proj_up(out)
        out = self.proj_drop(out)
        return out

class WindowTransformerBlock(nn.Module):
    """
    A non-shifted window transformer block that partitions the feature map into windows,
    applies window-based self-attention with an MLP, and then merges the windows back.
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, reduction_ratio=2, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = 0  # No shift in this block
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, reduction_ratio, dropout)
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
        x: Input tensor of shape (B, H, W, C).
        Returns:
            Tensor of the same shape.
        """
        B, H, W, C = x.shape
        shortcut = x
        # Normalize input before attention
        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)
        shifted_x = x  # No shift in this block

        # Partition input into windows
        windows = window_partition(shifted_x, self.window_size)
        B_win, num_windows, N, _ = windows.shape
        windows = windows.view(B_win * num_windows, N, C)

        # Apply window-based attention
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(B_win, num_windows, N, C)

        # Merge windows back into feature map
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Add residual connection and then apply MLP with normalization
        x = shortcut + shifted_x
        x = x + self.mlp(self.norm2(x.view(B, -1, C)).view(B, H, W, C))
        return x

class ShiftedWindowTransformerBlock(nn.Module):
    """
    A transformer block that shifts the window partitioning to allow cross-window interaction.
    After shifting, the windows are partitioned, attention is applied, and then the shift is reversed.
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, reduction_ratio=2, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2  # Shift by half window size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, reduction_ratio, dropout)
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
        x: Input tensor of shape (B, H, W, C).
        Returns:
            Tensor of the same shape, after shifted-window attention.
        """
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)
        # Apply shift to enable cross-window interactions
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition shifted feature map into windows
        windows = window_partition(shifted_x, self.window_size)
        B_win, num_windows, N, _ = windows.shape
        windows = windows.view(B_win * num_windows, N, C)

        # Apply attention inside windows
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(B_win, num_windows, N, C)

        # Merge windows back
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x.view(B, -1, C)).view(B, H, W, C))
        return x

class CoarseGlobalAttention(nn.Module):
    """
    Coarse global attention applies average pooling (to downsample), then self-attention 
    on the pooled tokens, and finally upsamples the output to add a global context.
    """
    def __init__(self, dim, num_heads, pool_size=4, qkv_bias=True, dropout=0.):
        super().__init__()
        self.pool_size = pool_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Input tensor of shape (B, H, W, C).
        Downsamples the input, computes attention on pooled features, upsamples the result,
        and adds it back to the original input.
        """
        B, H, W, C = x.shape
        shortcut = x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        pooled_x = self.pool(x)    # Downsample
        H_p, W_p = pooled_x.shape[2:]
        N_p = H_p * W_p
        pooled_x = pooled_x.flatten(2).transpose(1, 2)  # (B, N_p, C)
        qkv = self.qkv(pooled_x).reshape(B, N_p, 3, self.num_heads, C // self.num_heads)\
              .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N_p, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H_p, W_p)
        # Upsample to the original feature map size
        attn_out_upsampled = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        attn_out_upsampled = attn_out_upsampled.permute(0, 2, 3, 1)  # (B, H, W, C)
        return shortcut + attn_out_upsampled

class TransformerModel(nn.Module):
    """
    The main transformer-based upscaling model.
    
    Steps:
      1. A shallow CNN encoder extracts features.
      2. Patch embedding converts features into a grid of tokens.
      3. Tokens are padded and processed by a series of transformer blocks (windowed, shifted-window, and coarse attention).
      4. Patch unembedding reconstructs a feature map from tokens.
      5. A CNN decoder refines the features and predicts a residual image.
      6. Upsampling operations and a global residual connection produce the final upscaled image.
    """
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 35,
                 encoder_blocks: int = 3,
                 decoder_blocks: int = 3,
                 use_ca: bool = True,
                 transformer_dim: int = 84,
                 num_transformer_blocks: int = 6,
                 unembedding_mode: Literal['pixelshuffle', 'convtranspose'] = 'convtranspose',
                 num_heads: int = 2,
                 mlp_ratio: float = 1.5,
                 dropout: float = 0.1,
                 window_size: int = 8,
                 patch_stride: int = 8,
                 reduction_ratio: int = 2,
                 global_context_interval: int = 1,
                 coarse_attention_pool_size: int = 4):
        super(TransformerModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.patch_stride = patch_stride
        self.window_size = window_size

        # Encoder: Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        encoder_blocks = [ResBlock(base_channels, use_ca=use_ca) for _ in range(encoder_blocks)]
        self.encoder = nn.Sequential(*encoder_blocks)

        # Residual upscale branch to upsample the original feature map (for global residual connection)
        self.up1 = Upsampler(conv=default_conv, n_feats=base_channels)
        self.up1_conv = BasicConv(base_channels, 3, 3, 1, 1)

        # Patch embedding converts features to a token grid
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=patch_stride, stride=patch_stride)

        # Build the transformer blocks. Alternate between Window and ShiftedWindow blocks,
        # inserting CoarseGlobalAttention blocks based on global_context_interval.
        trans_blocks = []
        for i in range(num_transformer_blocks):
            if i % 2 == 0:
                trans_blocks.append(WindowTransformerBlock(transformer_dim, window_size, num_heads, mlp_ratio, reduction_ratio, dropout))
            else:
                trans_blocks.append(ShiftedWindowTransformerBlock(transformer_dim, window_size, num_heads, mlp_ratio, reduction_ratio, dropout))
            if (i + 1) % global_context_interval == 0 and i != num_transformer_blocks - 1:
                trans_blocks.append(CoarseGlobalAttention(transformer_dim, num_heads, coarse_attention_pool_size))
        self.transformer_blocks = nn.Sequential(*trans_blocks)

        # Patch unembedding reconstructs feature map from tokens
        if unembedding_mode == 'pixelshuffle':
            self.patch_unembed = nn.Sequential(
                nn.Conv2d(transformer_dim, base_channels * (self.patch_stride ** 2), kernel_size=1),
                nn.PixelShuffle(self.patch_stride),
                ResBlock(base_channels, use_ca=False)
            )
        elif unembedding_mode == 'convtranspose':
            self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels,
                                                     kernel_size=self.patch_stride, stride=self.patch_stride)
        else:
            raise ValueError("unembedding_mode must be either 'pixelshuffle' or 'convtranspose'")

        # Decoder refines the upscaled feature map
        decoder_blocks = [ResBlock(base_channels, use_ca=use_ca) for _ in range(decoder_blocks)]
        self.decoder = nn.Sequential(*decoder_blocks)
        self.final_decode = default_conv(base_channels, in_channels, 3)

        # Final upscale operation for the predicted residual
        self.final_upscale = Upsampler(conv=default_conv, n_feats=3)

    def forward(self, x: torch.Tensor, res_out: Tuple[int, int] = (1080, 1920),
                upscale_factor: int = None, require_ratio: bool = True) -> torch.Tensor:
        """
        Forward pass:
            - Upscale input, process through encoder, transform tokens via transformer blocks,
              and then decode and combine residuals.
              
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Desired output resolution.
            upscale_factor (int): Upscaling factor; if provided, overrides res_out.
            require_ratio (bool): Whether to enforce resizing to the target aspect ratio.
            
        Returns:
            torch.Tensor: Upscaled output image tensor.
        """
        # Determine target upscale dimensions
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)
        elif upscale_factor is None:
            upscale_factor = math.ceil(max(res_out[0] / x.shape[2], res_out[1] / x.shape[3]))

        # Encoder: extract features
        feat = self.relu(self.conv1(x))
        feat = self.encoder(feat)
        B, C, H_feat, W_feat = feat.shape

        # Residual upscale branch (process original feature for global residual)
        upscaled_input = self.up1(feat, upscale_factor)
        upscaled_input = self.up1_conv(upscaled_input)

        # Pad feature map so that its dimensions are multiples of patch_stride
        pad_h = (self.patch_stride - H_feat % self.patch_stride) % self.patch_stride
        pad_w = (self.patch_stride - W_feat % self.patch_stride) % self.patch_stride
        feat_pad = F.pad(feat, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else feat

        # Patch embedding: convert features to tokens
        tokens = self.patch_embed(feat_pad)  # (B, transformer_dim, H_t, W_t)
        B, C_t, H_t, W_t = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1).contiguous()  # (B, H_t, W_t, transformer_dim)

        # Pad token grid to ensure full windows
        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right = (self.window_size - W_t % self.window_size) % self.window_size
        orig_H, orig_W = H_t, W_t
        if pad_bottom or pad_right:
            tokens = tokens.permute(0, 3, 1, 2)
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom))
            tokens = tokens.permute(0, 2, 3, 1).contiguous()
            H_t, W_t = tokens.shape[1], tokens.shape[2]

        # Process tokens through transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        if pad_bottom or pad_right:
            tokens = tokens[:, :orig_H, :orig_W, :]

        tokens = tokens.permute(0, 3, 1, 2).contiguous()  # (B, transformer_dim, H_t, W_t)

        # Patch unembedding: convert tokens back to a feature map
        feat_trans = self.patch_unembed(tokens)
        feat_trans = feat_trans[:, :, :H_feat, :W_feat]   # Crop any extra padding

        # Combine with skip connection
        combined_feat = feat[:, :, :H_feat, :W_feat] + feat_trans

        # Decoder: refine the combined features and predict the residual
        dec = self.decoder(combined_feat)
        residual = self.final_decode(dec)

        # Upsample the residual prediction
        residual_up = self.final_upscale(residual, upscale_factor)

        # Final output: combine upscaled input and predicted residual
        out = upscaled_input + residual_up

        # Optionally, resize to match desired aspect ratio
        if require_ratio and res_out != (out.shape[2], out.shape[2]):
            hr_squash = transforms.Resize(res_out)
            out = hr_squash(out)

        return torch.clamp(out, 0.0, 1.0)

# Quick test block
if __name__ == "main":
    model = TransformerModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 720, 1280)
    start = time.time()
    output = model(dummy_input, upscale_factor=3)
    end = time.time()
    print("Output shape:", output.shape)  # Expected shape based on upscale factor
    print("Time taken:", round(end - start, 4), 's')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
