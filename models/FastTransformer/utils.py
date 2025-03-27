import torch
import torch.nn as nn
import math


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
