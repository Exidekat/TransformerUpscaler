import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
torch.autograd.set_detect_anomaly(True)

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
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
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
    Build once in __init__, then choose the sub-module at forward time.
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

class ARFB(nn.Module):
    def __init__(self, in_channels, reduction=2, groups=4):
        super(ARFB, self).__init__()
        reduced_channels = in_channels // reduction
        self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1, groups=groups, bias=True)
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        # Learnable scale for the residual branch
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        identity = x
        out = self.reduce(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.expand(out)
        return identity + self.scale * out
    
class HighFrequencyModule(nn.Module):
    def __init__(self, kernel_size=2):
        super(HighFrequencyModule, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # x shape: (B, C, H, W)
        smooth = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.kernel_size)
        smooth_upsampled = F.interpolate(smooth, size=x.shape[2:], mode='bilinear', align_corners=False)
        high_freq = x - smooth_upsampled
        return high_freq

class WindowAttention(nn.Module):
    """
    HAT-style efficient window attention.
    Query is kept at full resolution for quality, but for computing similarities
    we project q, k, and v to a lower-dimensional space. k and v are reduced via a shared
    reduction layer; q is reduced using a separate projection to match dimensions.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0, reduction: int = 2):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Assume square window: window_size x window_size.
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        # Separate linear layers for query and for key/value.
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        # Reduction for k and v:
        self.reduce = nn.Linear(self.head_dim, self.head_dim // reduction, bias=False)
        # For computing attention we reduce q as well—but we preserve the full q later.
        self.q_reduce = nn.Linear(self.head_dim, self.head_dim // reduction, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        # Final projection: output dimension returns to original dim.
        self.proj = nn.Linear(num_heads * (self.head_dim // reduction), dim)
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

    def forward(self, x):
        B, N, C = x.shape
        # Compute query.
        q = self.q_proj(x)  # (B, N, C)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        # For attention computation, reduce q:
        q_reduced = self.q_reduce(q)  # (B, num_heads, N, head_dim//reduction)

        # Compute key and value together.
        kv = self.kv_proj(x).view(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]  # each: (B, num_heads, N, head_dim)
        # Reduce k and v.
        k_reduced = self.reduce(k)  # (B, num_heads, N, head_dim//reduction)
        v_reduced = self.reduce(v)  # (B, num_heads, N, head_dim//reduction)

        # Compute scaled dot-product attention in the reduced dimension.
        attn = q_reduced @ k_reduced.transpose(-2, -1)  # (B, num_heads, N, N)
        # Add relative positional bias.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size * self.window_size,
                                                             self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_reduced  # (B, num_heads, N, head_dim//reduction)
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, num_heads * (head_dim//reduction))
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)   # shape [B, C, 1, 1]
        y = self.fc(y)         # shape [B, C, 1, 1]
        return x * y
    
class WindowTransformerBlock(nn.Module):
    """
    A non-shifted window transformer block:
      - window_size: size of the local window
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = 0
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

class EnhancedWindowTransformerBlock(nn.Module):
    """
    Combines window-based attention + channel attention + optional shift_size.
    """
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 shift_size=0):
        super().__init__()
        self.window_size = window_size
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

        # Channel attention after the main block
        self.channel_attention = ChannelAttention(dim)

    def forward(self, x):
        """
        x is shape (B, H, W, C). We'll do:
          1) possibly shift x
          2) partition into windows
          3) window attention
          4) reverse partition
          5) undo shift
          6) MLP + residual
          7) channel attention
        """
        B, H, W, C = x.shape
        shortcut = x

        # 1. LayerNorm over (B, H*W, C)
        x_ln = self.norm1(x.reshape(B, -1, C)).reshape(B, H, W, C)

        # 2. Cyclic shift if shift_size > 0
        if self.shift_size > 0:
            # We "roll" along height and width
            x_ln = torch.roll(
                x_ln,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)  # shift in H and W dimensions
            )

        # 3. Window partition
        windows = window_partition(x_ln, self.window_size)  # (B, num_windows, window_size^2, C)
        B_, num_wins, N, _ = windows.shape
        windows = windows.view(B_ * num_wins, N, C)  # flatten for attn

        # 4. Window attention
        attn_windows = self.attn(windows)  # shape (B_ * num_wins, N, C)
        attn_windows = attn_windows.view(B_, num_wins, N, C)

        # 5. Reverse window partition
        x_attn = window_reverse(attn_windows, self.window_size, H, W)

        # 6. Undo the cyclic shift if shift_size > 0
        if self.shift_size > 0:
            x_attn = torch.roll(
                x_attn,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )

        # 7. Residual + MLP
        x_out = shortcut + x_attn
        # MLP
        x_mlp_ln = self.norm2(x_out.reshape(B, -1, C)).reshape(B, H, W, C)
        mlp_out = self.mlp(x_mlp_ln)
        x_out = x_out + mlp_out

        # 8. Channel attention
        # Switch to (B, C, H, W) for ChannelAttention, then switch back
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        x_out = self.channel_attention(x_out)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()

        return x_out
    
class TransformerModel(nn.Module):
    def __init__(self, in_channels=3,
                 base_channels=48,
                 transformer_dim=128,
                 num_window_blocks=4,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 window_size=8):
        super(TransformerModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)  
        self.conv2 = ARFB(base_channels, reduction=2)
        self.relu = nn.ReLU(inplace=True)
        
        # High-Frequency Module to capture texture details
        self.hf_module = HighFrequencyModule(kernel_size=2)
        self.hf_fusion_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1, bias=True)
        
        # Residual upscale branch
        self.up1 = Upsampler(conv=default_conv, n_feats=base_channels)
        self.up1_conv = BasicConv(base_channels, 3, 3, 1, 1)
        
        # Final upscale branch 
        self.final_upscale = Upsampler(conv=default_conv, n_feats=3)
        self.final_upscale_conv = default_conv(3, 3, 3)
        
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=8, stride=8)
        self.window_size = window_size
        blocks = []
        for i in range(num_window_blocks):
            shift = self.window_size // 2 if (i % 2 == 1) else 0
            block = EnhancedWindowTransformerBlock(
                dim=transformer_dim,
                window_size=self.window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                shift_size=shift
            )
            blocks.append(block)

        self.window_blocks = nn.ModuleList(blocks)
        
        # Patch unembedding
        self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels, kernel_size=8, stride=8)
        
        # CNN decoder refinements
        self.decoder_block1 = ARFB(base_channels, reduction=2)
        self.decoder_block2 = ARFB(base_channels, reduction=2) 
        self.decoder_block3 = ARFB(base_channels, reduction=2) 
        self.decoder_conv_final = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, res_out=(1080, 1920), upscale_factor=None, require_ratio=True):
        
        if upscale_factor is not None:
            res_out = (x.shape[2] * upscale_factor, x.shape[3] * upscale_factor)
        elif upscale_factor is None:
            upscale_factor = math.ceil(max(res_out[0] / x.shape[2], res_out[1] / x.shape[3]))
        
        # Shallow feature extraction with ARFB
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        
        # Extract high-frequency details
        hf = self.hf_module(feat)
        fused_feat_cat = torch.cat([feat, hf], dim=1)
        fused_feat = self.relu(self.hf_fusion_conv(fused_feat_cat))
        H_fused, W_fused = fused_feat.shape[2], fused_feat.shape[3]
        pad_h_fused = (8 - H_fused % 8) % 8
        pad_w_fused = (8 - W_fused % 8) % 8
        fused_feat_padded = F.pad(fused_feat, (0, pad_w_fused, 0, pad_h_fused), mode='reflect') if (pad_h_fused or pad_w_fused) else fused_feat
        
        
        tokens = self.patch_embed(fused_feat_padded)  # (B, transformer_dim, H_t, W_t)
        B, C_t, H_t, W_t = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1).contiguous()  # (B, H_t, W_t, transformer_dim)
        
        # Pad token grid for window partitioning.
        B, H_t, W_t, _ = tokens.shape
        pad_bottom = (self.window_size - H_t % self.window_size) % self.window_size
        pad_right  = (self.window_size - W_t % self.window_size) % self.window_size
        if pad_bottom or pad_right:
            # Permute to (B, transformer_dim, H_t, W_t) for padding
            tokens = tokens.permute(0, 3, 1, 2).contiguous()
            tokens = F.pad(tokens, (0, pad_right, 0, pad_bottom), mode='reflect')
            # Permute back to (B, H, W, transformer_dim)
            tokens = tokens.permute(0, 2, 3, 1).contiguous()
        
        # Process through enhanced transformer blocks
        for block in self.window_blocks:
            tokens = block(tokens)
        
        tokens = tokens.permute(0, 3, 1, 2).contiguous()
        feat_trans = self.patch_unembed(tokens)
        feat_trans = feat_trans[:, :, :fused_feat.shape[2], :fused_feat.shape[3]]
        
        # Decoder to predict residual
        dec = self.relu(self.decoder_block1(feat_trans))
        dec = self.relu(self.decoder_block2(dec))
        dec = self.relu(self.decoder_block3(dec))
        residual = self.decoder_conv_final(dec)
        
        residual_up = self.final_upscale(residual, upscale_factor)
        residual_up = self.final_upscale_conv(residual_up)
        
        upscaled_input = self.up1(feat, upscale_factor)  # using original feat for skip
        upscaled_input = self.up1_conv(upscaled_input)
        
        out = upscaled_input + residual_up
        
        if require_ratio and res_out != (out.shape[2], out.shape[3]):
            hr_squash = transforms.Resize(res_out)
            out = hr_squash(out)
        
        return torch.clamp(out, 0.0, 1.0)
    
# Quick test.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 100, 100)
    output = model(dummy_input, upscale_factor=6)
    print("Output shape:", output.shape)  # Expected: (1, 3, 600, 600)