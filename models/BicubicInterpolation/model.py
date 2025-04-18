#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TransformerModel(nn.Module):
    """
    Fake model for testing purposes.

    Outputs bicubic interpolation of input
    """
    def __init__(self):
        super(TransformerModel, self).__init__()

    def forward(self,
                x: torch.Tensor,
                res_out: Tuple[int, int] = (1080, 1920),
                upscale_factor: int = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Target output resolution (height, width).
            upscale_factor (int, optional): If provided, overrides res_out by scaling input spatial dims.

        Returns:
            torch.Tensor: Upscaled image of shape (B, 3, target_H, target_W).
        """
        # Override target resolution if upscale_factor is given
        if upscale_factor is not None:
            _, _, h, w = x.shape
            res_out = (h * upscale_factor, w * upscale_factor)
        # Just provide an upscale interpolation
        upscaled_input = F.interpolate(x, size=res_out, mode='bicubic', align_corners=False)

        return upscaled_input

# Quick test.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)
