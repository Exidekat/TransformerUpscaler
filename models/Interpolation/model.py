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

    def forward(self, x: torch.Tensor, res_out: Tuple[int, int] = (1080, 1920)) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            res_out (Tuple[int, int]): Target output resolution (height, width).

        Returns:
            torch.Tensor: Upscaled image of shape (B, 3, target_H, target_W).
        """
        # Just provide an upscale interpolation
        upscaled_input = F.interpolate(x, size=res_out, mode='bicubic', align_corners=False)

        return upscaled_input

# Quick test.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)
