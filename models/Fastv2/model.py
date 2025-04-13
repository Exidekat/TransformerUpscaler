#!/usr/bin/env python
"""
TransformerModel.py

Revised Transformer-based model for image upscaling using relative positional encoding.

Architecture Overview:

A shallow CNN encoder extracts features from the low-resolution input.

A convolutional patch embedding converts the feature map into a grid of tokens.

The token grid is partitioned into non-overlapping windows.

A series of window transformer blocks (with relative positional encoding)
process the tokens.

The windows are merged back into a 2D token grid.

A patch unembedding layer (transposed convolution) converts tokens back to a feature map.

A CNN decoder refines the features and predicts a residual image which is upscaled with sub pixel convolutions.

A global residual connection adds the predicted residual to a sub pixel convolution upscaled input.

This design is resolution-agnostic because the patch embedding and window operations
handle arbitrary input sizes, and the global residual is computed via bicubic upsampling.
A minor spatial mismatch (if dimensions are not divisible by the window size) is resolved via cropping.
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

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                        padding=(kernel_size // 2), bias=bias, groups=groups)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
    bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding,
            dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU(inplace=True) if relu else None
            self.up_size = up_size
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
    Allows multiple (fixed) integer scale factors with sub-pixel convolution.
    Build once in init, then choose the sub-module at forward time.
    """

    def __init__(self, conv, n_feats, valid_scales=(2, 3, 4, 6),
                bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        self.upsamplers = nn.ModuleDict()

        for scale in valid_scales:
            # Build a sequence of conv + pixelshuffle blocks for this 'scale'
            blocks = []
            if (scale & (scale - 1)) == 0:
                # scale is a power of two (e.g. 2,4,8,...)
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

            # Register as a sub-module keyed by the integer scale
            self.upsamplers[str(scale)] = nn.Sequential(*blocks)

    def forward(self, x, scale):
        # scale should be one of the scales in valid_scales
        scale_str = str(scale)
        if scale_str not in self.upsamplers:
            raise ValueError(f"Requested scale={scale} was not built.")
        return self.upsamplers[scale_str](x)

class CALayer(nn.Module):
    """
    RCAN's Channel Attention (CA) layer
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # Global Average Pooling is done on (B, C, H, W) -> (B, C, 1, 1)
        # Then pass through conv->ReLU->conv->Sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # "Excitation" step: reduce then re-expand channels
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, C, H, W)
        y = self.avg_pool(x)    # (B, C, 1, 1)
        y = self.conv_du(y)     # (B, C, 1, 1)
        return x * y            # Elementwise multiply

class ResBlock(nn.Module):
    # Lightweight Residual Block definition
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
        res += x
        return res

class WindowAttention(nn.Module):
    """
    Lightweight Window-based multi-head self-attention (LW-MSA) inspired by ESRT,
    with relative positional encoding.

    Attention calculation is performed in a reduced dimension for efficiency.
    """
    def __init__(self,
                dim: int,
                window_size: int,
                num_heads: int,
                reduction_ratio: int = 2, # Ratio to reduce dimension for attention
                dropout: float = 0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        # Calculate reduced dimension, ensure divisibility
        self.reduced_dim = dim // reduction_ratio
        assert self.reduced_dim * reduction_ratio == dim, f"dim ({dim}) must be divisible by reduction_ratio ({reduction_ratio})"
        assert self.reduced_dim % num_heads == 0, f"reduced_dim ({self.reduced_dim}) must be divisible by num_heads ({num_heads})"

        self.head_dim = self.reduced_dim // num_heads # Head dim is based on reduced dim
        self.scale = self.head_dim ** -0.5

        # 1. Project input down to reduced dimension
        self.proj_down = nn.Linear(dim, self.reduced_dim, bias=True)

        # 2. QKV projection from reduced dimension
        self.qkv = nn.Linear(self.reduced_dim, self.reduced_dim * 3, bias=True)

        # 3. Attention dropout
        self.attn_drop = nn.Dropout(dropout)

        # 4. Project output back up to original dimension
        self.proj_up = nn.Linear(self.reduced_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout) # Dropout after final projection

        # --- Relative Positional Bias (Same as before) ---
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
        # --- End Relative Positional Bias ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C=dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C=dim).
        """
        B, N, C = x.shape
        assert C == self.dim, "Input dimension C doesn't match module's dim"

        # 1. Project down
        x_reduced = self.proj_down(x) # (B, N, reduced_dim)

        # 2. Get QKV from reduced dimension
        # (B, N, 3 * reduced_dim) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x_reduced).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each: (B, num_heads, N, head_dim)

        # 3. Attention calculation (in reduced dimension head_dim)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, N, N)

        # Add relative positional bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0) # (1, num_heads, N, N)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum with V
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        out = torch.matmul(attn, v)

        # Reshape and combine heads
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, reduced_dim)
        out = out.transpose(1, 2).reshape(B, N, self.reduced_dim)

        # 4. Project back up to original dimension
        out = self.proj_up(out) # (B, N, dim)
        out = self.proj_drop(out)

        return out
    
class WindowTransformerBlock(nn.Module):
    """
    A non-shifted window transformer block:
    - window_size: size of the local window
    - shift_size=0 (no shift)
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, reduction_ratio=2, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = 0
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
        x: (B, H, W, C)
        Returns: (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x

        # 1. Layer norm
        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)

        # 2. No shift here
        shifted_x = x

        # 3. Window partition
        windows = window_partition(shifted_x, self.window_size)  
        B_win, num_windows, N, _ = windows.shape
        windows = windows.view(B_win * num_windows, N, C)

        # 4. Window attention
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(B_win, num_windows, N, C)

        # 5. Reverse partition
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 6. Residual + MLP
        x = shortcut + shifted_x
        x = x + self.mlp(self.norm2(x.view(B, -1, C)).view(B, H, W, C))

        return x
    
class ShiftedWindowTransformerBlock(nn.Module):
    """
    A shifted window transformer block:
    - window_size: size of the local window
    - shift_size=window_size//2 for cross-window interaction
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, reduction_ratio=2, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
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
        x: (B, H, W, C)
        Returns: (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x

        # 1. Layer norm
        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)

        # 2. Shift feature map
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # 3. Window partition
        windows = window_partition(shifted_x, self.window_size)
        B_win, num_windows, N, _ = windows.shape
        windows = windows.view(B_win * num_windows, N, C)

        # 4. Window attention
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(B_win, num_windows, N, C)

        # 5. Reverse partition
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 6. Reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # 7. Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.view(B, -1, C)).view(B, H, W, C))
        return x
    
class CoarseGlobalAttention(nn.Module):
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
        # Upsampling needs target size or scale factor - calculate in forward

    def forward(self, x):
        # Input x: (B, H, W, C)
        B, H, W, C = x.shape
        shortcut = x
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)

        # Downsample
        pooled_x = self.pool(x) # (B, C, H // pool_size, W // pool_size)
        H_p, W_p = pooled_x.shape[2:]
        N_p = H_p * W_p
        pooled_x = pooled_x.flatten(2).transpose(1, 2) # (B, N_p, C)

        # Attention on pooled features
        qkv = self.qkv(pooled_x).reshape(B, N_p, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out = (attn @ v).transpose(1, 2).reshape(B, N_p, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out) # (B, N_p, C)

        # Reshape and Upsample
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H_p, W_p)
        attn_out_upsampled = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False) # (B, C, H, W)

        attn_out_upsampled = attn_out_upsampled.permute(0, 2, 3, 1) # (B, H, W, C)

        # Add back to shortcut
        return shortcut + attn_out_upsampled

class TransformerModel(nn.Module):
    """
    Transformer-based model for image upscaling using relative positional encoding.

    Architecture:
    1. CNN encoder extracts features from the low-resolution input.
    2. Patch embedding converts features to a grid of tokens.
    3. Tokens are partitioned into non-overlapping windows.
    4. Window transformer blocks process the tokens.
    5. Windows are merged back into a token grid.
    6. Patch unembedding reconstructs a feature map from tokens.
    7. CNN + Pixel Shuffle operations upscale the feature map.
    8. CNN + Pixel Shuffle operations create a residual from the input.
    9. Global residual connection: The predicted residual is added to the upscaled input.
    """
    def __init__(
        self,
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
        coarse_attention_pool_size: int = 4  
        ):
        super(TransformerModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.patch_stride = patch_stride
        self.window_size = window_size
        
        # Encoder: Shallow CNN.
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        encoder_blocks = [
            ResBlock(base_channels, use_ca=use_ca) for _ in range(encoder_blocks)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)
            
        # Residual upscale.
        self.up1 = Upsampler(conv=default_conv, n_feats=base_channels)
        self.up1_conv = BasicConv(base_channels, 3, 3, 1, 1)
        
        # Patch embedding: converts feature map to tokens.
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=patch_stride, stride=patch_stride)

        # Window transformer blocks.
        trans_blocks = []
        for i in range(num_transformer_blocks):
            if i % 2 == 0:
                trans_blocks.append(WindowTransformerBlock(transformer_dim, window_size, num_heads, 
                                                                mlp_ratio, reduction_ratio, dropout))
            else:
                trans_blocks.append(ShiftedWindowTransformerBlock(transformer_dim, window_size, num_heads, 
                                                                        mlp_ratio, reduction_ratio, dropout))   
            
            if (i + 1) % global_context_interval == 0 and i != num_transformer_blocks - 1:
                trans_blocks.append(CoarseGlobalAttention(transformer_dim, num_heads, coarse_attention_pool_size))
                
        self.transformer_blocks = nn.Sequential(*trans_blocks) 

        # Patch unembedding: converts tokens back to a feature map.
        if unembedding_mode == 'pixelshuffle': 
            self.patch_unembed = nn.Sequential(
                nn.Conv2d(transformer_dim, base_channels * (self.patch_stride**2), kernel_size=1),
                nn.PixelShuffle(self.patch_stride),
                ResBlock(base_channels, use_ca=False)
            )
        elif unembedding_mode == 'convtranspose':
            self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels,
                                                    kernel_size=self.patch_stride, stride=self.patch_stride)
        
        else:
            raise ValueError("unembedding_mode must be either 'pixelshuffle' or 'convtranspose'")
        
        # Decoder: CNN.
        decoder_blocks = [
            ResBlock(base_channels, use_ca=use_ca) for _ in range(decoder_blocks)  
        ]   
        self.decoder = nn.Sequential(*decoder_blocks)
        self.final_decode = default_conv(base_channels, in_channels, 3)
        
        # Final upscale.
        self.final_upscale = Upsampler(conv=default_conv, n_feats=3)  

    def forward(self, x: torch.Tensor, res_out: Tuple[int, int] = (1080, 1920), upscale_factor: int = None, require_ratio: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Target output resolution (height, width).
            upscale_factor (int): Upscale factor (optional, overrides 'res_out').
            require_ratio (bool): whether to require that the upscaled image be resized down to the target resolution.

        Returns:
            torch.Tensor: Upscaled image of shape (B, 3, target_H, target_W).
        """
        # Compute target upscale.
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)
        elif upscale_factor is None:
            upscale_factor = math.ceil(max(res_out[0] / x.shape[2], res_out[1] / x.shape[3]))

        # Encoder.
        feat = self.relu(self.conv1(x))
        feat = self.encoder(feat)
        B, C, H_feat, W_feat = feat.shape
        
        # Residual upscale branch (using the original feat).
        upscaled_input = self.up1(feat, upscale_factor)
        upscaled_input = self.up1_conv(upscaled_input)

        # --- Pad 'feat' so that its spatial dims are multiples of 8 ---
        pad_h = (self.patch_stride - H_feat % self.patch_stride) % self.patch_stride
        pad_w = (self.patch_stride - W_feat % self.patch_stride) % self.patch_stride
        if pad_h or pad_w:
            feat_pad = F.pad(feat, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            feat_pad = feat

        # --- Patch embedding on padded feature map ---
        tokens = self.patch_embed(feat_pad)  # (B, transformer_dim, H_t, W_t)
        B, C_t, H_t, W_t = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1).contiguous()  # (B, H_t, W_t, transformer_dim)

        # Pad token grid for window partitioning.
        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right  = (self.window_size - W_t % self.window_size) % self.window_size
        orig_H, orig_W = H_t, W_t
        if pad_bottom or pad_right:
            tokens = tokens.permute(0, 3, 1, 2)  # (B, transformer_dim, H_t, W_t)
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom))
            tokens = tokens.permute(0, 2, 3, 1).contiguous()
            H_t, W_t = tokens.shape[1], tokens.shape[2]

        # Process windows with transformer blocks.
        for block in self.transformer_blocks:
            tokens = block(tokens)
            
        # Remove padding added for window partitioning.
        if pad_bottom or pad_right:
            tokens = tokens[:, :orig_H, :orig_W, :]

        tokens = tokens.permute(0, 3, 1, 2).contiguous()  # (B, transformer_dim, H_t, W_t)

        # --- Patch unembedding ---
        feat_trans = self.patch_unembed(tokens)  # (B, base_channels, H_pad, W_pad)

        # Crop feat_trans to remove the padding from before patch embedding.
        feat_trans = feat_trans[:, :, :H_feat, :W_feat]   

        # Combine skip connection.
        feat = feat[:, :, :H_feat, :W_feat]
        combined_feat = feat + feat_trans

        # Decoder.
        dec = self.decoder(combined_feat)
        residual = self.final_decode(dec)


        # Upsample the predicted residual.
        residual_up = self.final_upscale(residual, upscale_factor)

        # Final output.
        out = upscaled_input + residual_up

        # Downsize if the upscale factor over shoots the desired aspect ratio
        if require_ratio and res_out != (out.shape[2], out.shape[2]):
            hr_squash = transforms.Resize(res_out)
            out = hr_squash(out)

        return torch.clamp(out, 0.0, 1.0)

# Quick test.
if __name__ == "main":
    model = TransformerModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 720, 1280)
    start = time.time()
    output = model(dummy_input, upscale_factor=3)
    end = time.time()
    print("Output shape:", output.shape)  # Expected: (1, 3, 600, 600)
    print("Time taken:", round(end - start, 4), 's')  # Measure time taken for the forward pass.
    print("Model parameters:", sum(p.numel() for p in model.parameters()))