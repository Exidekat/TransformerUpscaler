#!/usr/bin/env python
"""
model.py

This model performs image upscaling using a transformer architecture modified
to integrate insights from Local Attribution Mapping (LAM). In this version,
the transformer branch is replaced with a hierarchical stack of LAM blocks –
each of which uses overlapping local attention windows to bridge neighboring
regions (OCAB) and a channel attention branch (CAB) to incorporate global channel
correlations. The design recognizes that the actual range of pixels contributing
to the output is often much smaller than the theoretical receptive field; hence,
the module explicitly aggregates and fuses the effective pixels and is controlled
by hyperparameters such as the overlapping ratio and CAB weighting factor.

Overall pipeline:
  1. A shallow CNN encoder extracts features from the input image.
  2. Patch embedding converts features into a grid of tokens.
  3. A hierarchical stack of LAMTransformerBlocks processes tokens.
  4. Patch unembedding reconstructs a feature map from tokens.
  5. A CNN decoder refines the result, and an upsample branch (with pixel-shuffle)
     along with a global residual connection produces the final upscaled image.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple, Literal
import time

# Apply weight normalization to modules.
wn = lambda x: torch.nn.utils.parametrizations.weight_norm(x)


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


def overlap_window_partition(x: torch.Tensor, window_size: int, gamma: float) -> Tuple[torch.Tensor, Tuple]:
    """
    Partitions the input tensor into overlapping windows.

    Given a base window size and an overlapping ratio gamma (e.g., 0.5),
    the effective window size is computed as:
        effective_window_size = int(window_size * (1 + gamma))
    The windows are extracted with a stride equal to the base window size.

    To ensure that the unfolded patches have size equal to effective_window_size,
    extra padding is applied if the input height or width is smaller than the effective size.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Base non-overlapping window size.
        gamma (float): Overlapping ratio (e.g., 0.5 implies 50% overlap).

    Returns:
        Tuple[torch.Tensor, Tuple]: A tuple containing:
           - windows: Tensor of shape (B, num_windows, effective_window_area, C).
           - info: A tuple (B, C, H_pad, W_pad, effective_window_size, stride) needed for reversal.
    """
    B, H, W, C = x.shape
    stride = window_size
    effective_win = int(window_size * (1 + gamma))

    # Permute to (B, C, H, W) for processing with unfold.
    x_perm = x.permute(0, 3, 1, 2)

    # Pad so that padded height/width are at least effective_win.
    pad_h = max(0, effective_win - H) if H < effective_win else ((stride - H % stride) % stride)
    pad_w = max(0, effective_win - W) if W < effective_win else ((stride - W % stride) % stride)

    x_pad = F.pad(x_perm, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = x_pad.shape
    patches = x_pad.unfold(2, effective_win, stride).unfold(3, effective_win, stride)
    # patches shape: (B, C, num_h, num_w, effective_win, effective_win)
    num_h = patches.shape[2]
    num_w = patches.shape[3]
    windows = patches.contiguous().view(B, C, num_h * num_w, effective_win * effective_win)
    windows = windows.permute(0, 2, 3, 1)  # (B, num_windows, effective_win*effective_win, C)
    return windows, (B, C, H_pad, W_pad, effective_win, stride)


def overlap_window_reverse(windows: torch.Tensor, output_size: Tuple[int, int], info: Tuple) -> torch.Tensor:
    """
    Reconstructs the original tensor from overlapping windows.

    The overlapping regions are averaged during reconstruction.

    Args:
        windows (torch.Tensor): Tensor of shape (B, num_windows, effective_area, C).
        output_size (Tuple[int, int]): Target (H, W) of the original (unpadded) tensor.
        info (Tuple): Information returned by overlap_window_partition:
                     (B, C, H_pad, W_pad, effective_win, stride).

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, H, W, C) where overlapping regions are averaged.
    """
    B, C, H_pad, W_pad, effective_win, stride = info
    num_h = H_pad // stride
    num_w = W_pad // stride

    # Reshape windows to (B, C, num_h, num_w, effective_win, effective_win)
    windows = windows.permute(0, 3, 1, 2).contiguous()  # (B, C, num_windows, effective_area)
    windows = windows.view(B, C, num_h, num_w, effective_win, effective_win)

    # Initialize output accumulator and a weight mask.
    output = torch.zeros(B, C, H_pad, W_pad, device=windows.device)
    weight = torch.zeros(B, C, H_pad, W_pad, device=windows.device)

    for i in range(num_h):
        for j in range(num_w):
            h_start = i * stride
            w_start = j * stride
            output[:, :, h_start:h_start + effective_win, w_start:w_start + effective_win] += windows[:, :, i, j]
            weight[:, :, h_start:h_start + effective_win, w_start:w_start + effective_win] += 1.0
    output = output / weight
    output = output[:, :, :output_size[0], :output_size[1]]
    output = output.permute(0, 2, 3, 1)
    return output


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    """
    Returns a 2D convolution with weight normalization.
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias, groups=groups)
    return wn(conv)


class BasicConv(nn.Module):
    """
    A basic convolutional (or transposed convolution) block that may include batch norm,
    activation, and an optional upsampling operation.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1,
                 dilation=1, groups=1, relu=True, bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = wn(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation,
                                              groups=groups, bias=bias))
        else:
            self.conv = wn(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias))
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
    Upsampler using sub-pixel convolution (PixelShuffle) for fixed scale factors.
    Builds a dictionary of upsampling modules keyed by the scale factor.
    """

    def __init__(self, conv, n_feats, valid_scales=(2, 3, 4, 6), bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
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
            raise ValueError(f"Requested scale={scale} was not built.")
        return self.upsamplers[scale_str](x)


class CALayer(nn.Module):
    """
    Channel Attention (CA) layer. Applies global average pooling followed by a bottleneck with 1x1 convolutions
    and a sigmoid activation, then reweights the input feature channels.
    """

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    """
    Residual block comprising two convolution layers, an activation, and optional channel attention.
    A residual connection adds the block's output to its input.
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
        return x + res


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative positional encoding.
    Projects input to a reduced dimension for efficiency and computes attention within each window.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, reduction_ratio: int = 2, dropout: float = 0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        self.reduced_dim = dim // reduction_ratio
        assert self.reduced_dim * reduction_ratio == dim, f"dim ({dim}) must be divisible by reduction_ratio ({reduction_ratio})"
        assert self.reduced_dim % num_heads == 0, f"reduced_dim ({self.reduced_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.reduced_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_down = nn.Linear(dim, self.reduced_dim, bias=True)
        self.qkv = nn.Linear(self.reduced_dim, self.reduced_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_up = nn.Linear(self.reduced_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

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
        B, N, C = x.shape
        assert C == self.dim, "Input dimension mismatch"
        x_reduced = self.proj_down(x)
        qkv = self.qkv(x_reduced).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.reduced_dim)
        out = self.proj_up(out)
        out = self.proj_drop(out)
        return out


class LAMTransformerBlock(nn.Module):
    """
    LAMTransformerBlock implements a transformer block inspired by Local Attribution Mapping (LAM).
    It extracts local information using overlapping window self-attention (OCAB style) and integrates
    global channel attention (CAB). The block consists of:
      - Layer normalization,
      - Overlapping window partitioning to extract patches,
      - Self-attention within each overlapping window,
      - A channel attention branch via CALayer,
      - Weighted fusion of the attention outputs,
      - A residual connection and an MLP to further process the features.

    This design promotes effective utilization of the pixels that truly contribute to SR,
    as seen in LAM visualizations.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 mlp_ratio: float = 4.0, gamma: float = 0.5, cab_weight: float = 0.01, dropout: float = 0.1):
        super(LAMTransformerBlock, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.gamma = gamma  # overlapping ratio
        self.num_heads = num_heads

        self.eff_win = int(window_size * (1 + gamma))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = WindowAttention(dim, self.eff_win, num_heads, reduction_ratio=2, dropout=dropout)
        self.ca = CALayer(dim, reduction=16)
        self.cab_weight = nn.Parameter(torch.tensor(cab_weight), requires_grad=True)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map of shape (B, H, W, C).

        Returns:
            torch.Tensor: Output feature map of shape (B, H, W, C) after overlapping window attention,
                          channel attention fusion, residual connection, and MLP.
        """
        B, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm1(x.view(B, -1, C)).view(B, H, W, C)

        # Partition into overlapping windows.
        windows, pack_info = overlap_window_partition(x_norm, self.window_size, self.gamma)
        # windows: (B, num_windows, eff_win*eff_win, C)
        num_windows = windows.shape[1]
        windows = windows.view(B * num_windows, self.eff_win * self.eff_win, C)
        # Apply self-attention to windows.
        windows_attn = self.attn(windows)
        windows_attn = windows_attn.view(B, num_windows, self.eff_win * self.eff_win, C)
        # Reconstruct feature map by averaging overlapping regions.
        attn_out = overlap_window_reverse(windows_attn, (H, W), pack_info)

        # Apply channel attention.
        x_ca = x_norm.permute(0, 3, 1, 2)
        ca_out = self.ca(x_ca)
        ca_out = ca_out.permute(0, 2, 3, 1)

        # Fuse branches with learnable CAB weighting.
        fused = attn_out + self.cab_weight * ca_out
        x_fused = shortcut + fused
        x_out = x_fused + self.mlp(self.norm2(x_fused.view(B, -1, C)).view(B, H, W, C))
        return x_out


def default_mlp(dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
    """
    Returns a simple two-layer MLP with GELU activation and dropout.
    """
    hidden_dim = int(dim * mlp_ratio)
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )


class TransformerModel(nn.Module):
    """
    The main transformer-based upscaling model modified with hierarchical Local Attention Mapping (LAM).

    Pipeline:
      1. A shallow CNN encoder extracts features.
      2. A residual upscale branch upscales original features (global residual).
      3. Patch embedding converts features into a token grid.
      4. A hierarchical stack of LAMTransformerBlocks processes the tokens.
      5. Patch unembedding reconstructs a feature map from tokens.
      6. A CNN decoder refines the upscaled feature map.
      7. Final upsampling (via pixel-shuffle) and a global residual connection produce the output HR image.

    The transformer branch now consists solely of LAM blocks that explicitly model effective pixel usage
    via overlapping windows and integrated channel attention.
    """

    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 35,
                 encoder_blocks: int = 3,
                 decoder_blocks: int = 3,
                 use_ca: bool = True,
                 transformer_dim: int = 84,
                 num_transformer_blocks: int = 6,
                 overlap_ratio: float = 0.5,
                 cab_weight: float = 0.01,
                 mlp_ratio: float = 1.5,
                 dropout: float = 0.1,
                 window_size: int = 8,
                 patch_stride: int = 8,
                 unembedding_mode: Literal['pixelshuffle', 'convtranspose'] = 'convtranspose'):
        super(TransformerModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.patch_stride = patch_stride
        self.window_size = window_size

        # Encoder: extract features with CNN.
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        encoder_layers = [ResBlock(base_channels, use_ca=use_ca) for _ in range(encoder_blocks)]
        self.encoder = nn.Sequential(*encoder_layers)

        # Residual upscale branch (global residual).
        self.up1 = Upsampler(conv=default_conv, n_feats=base_channels)
        self.up1_conv = BasicConv(base_channels, 3, 3, 1, 1)

        # Patch embedding: convert features to tokens.
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=patch_stride, stride=patch_stride)

        # Hierarchical stack of LAMTransformerBlocks.
        lam_blocks = []
        for _ in range(num_transformer_blocks):
            lam_blocks.append(LAMTransformerBlock(transformer_dim, window_size, num_heads=3,
                                                  mlp_ratio=mlp_ratio, gamma=overlap_ratio,
                                                  cab_weight=cab_weight, dropout=dropout))
        self.transformer_blocks = nn.Sequential(*lam_blocks)

        # Patch unembedding: convert tokens back to feature map.
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

        # Decoder: refine features.
        decoder_layers = [ResBlock(base_channels, use_ca=use_ca) for _ in range(decoder_blocks)]
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_decode = default_conv(base_channels, in_channels, 3)
        self.final_upscale = Upsampler(conv=default_conv, n_feats=3)

    def forward(self, x: torch.Tensor, res_out: Tuple[int, int] = (1080, 1920),
                upscale_factor: int = None, require_ratio: bool = True) -> torch.Tensor:
        """
        Forward pass:
          - Extract features via encoder.
          - Compute global residual via an upscaled branch.
          - Embed features into tokens.
          - Process tokens using the hierarchical LAM block stack.
          - Unembed tokens back to feature map.
          - Fuse with encoder features, refine via decoder, and predict residual.
          - Upsample the residual and combine with upscaled input.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Desired output resolution.
            upscale_factor (int): Upscaling factor; if provided, overrides res_out.
            require_ratio (bool): Whether to enforce resizing to the target aspect ratio.

        Returns:
            torch.Tensor: Upscaled output image tensor.
        """
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)
        elif upscale_factor is None:
            upscale_factor = math.ceil(max(res_out[0] / x.shape[2], res_out[1] / x.shape[3]))

        feat = self.relu(self.conv1(x))
        feat = self.encoder(feat)
        B, C, H_feat, W_feat = feat.shape

        upscaled_input = self.up1(feat, upscale_factor)
        upscaled_input = self.up1_conv(upscaled_input)

        pad_h = (self.patch_stride - H_feat % self.patch_stride) % self.patch_stride
        pad_w = (self.patch_stride - W_feat % self.patch_stride) % self.patch_stride
        feat_pad = F.pad(feat, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else feat

        tokens = self.patch_embed(feat_pad)
        B, C_t, H_t, W_t = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1).contiguous()

        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right = (self.window_size - W_t % self.window_size) % self.window_size
        orig_H, orig_W = H_t, W_t
        if pad_bottom or pad_right:
            tokens = tokens.permute(0, 3, 1, 2)
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom))
            tokens = tokens.permute(0, 2, 3, 1).contiguous()
            H_t, W_t = tokens.shape[1], tokens.shape[2]

        tokens = self.transformer_blocks(tokens)
        if pad_bottom or pad_right:
            tokens = tokens[:, :orig_H, :orig_W, :]
        tokens = tokens.permute(0, 3, 1, 2).contiguous()

        feat_trans = self.patch_unembed(tokens)
        feat_trans = feat_trans[:, :, :H_feat, :W_feat]

        combined_feat = feat[:, :, :H_feat, :W_feat] + feat_trans
        dec = self.decoder(combined_feat)
        residual = self.final_decode(dec)
        residual_up = self.final_upscale(residual, upscale_factor)

        out = upscaled_input + residual_up

        if require_ratio and res_out != (out.shape[2], out.shape[3]):
            hr_squash = transforms.Resize(res_out)
            out = hr_squash(out)

        return torch.clamp(out, 0.0, 1.0)


# Quick test block
if __name__ == "__main__":
    model = TransformerModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 720, 1280)
    start = time.time()
    output = model(dummy_input, upscale_factor=3, require_ratio=False)
    end = time.time()
    print("Output shape:", output.shape)
    print("Time taken:", round(end - start, 4), 's')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
