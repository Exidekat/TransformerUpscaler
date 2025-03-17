"""
TransformerModel.py

Revised Transformer-based model for image upscaling.

This architecture incorporates:
  - A deeper CNN encoder/decoder with residual connections.
  - A global residual connection: The network predicts a residual image that is added to a bicubic upscaled input.
  - A transformer module for global context that now operates on a lower number of tokens
    (using a larger patch size) to mitigate excessive memory allocation on MPS.

Input: Low resolution image (B, 3, 720, 1280)
Output: Upscaled image (B, 3, 1080, 1920)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    A transformer block that applies multi-head self-attention
    and a feed-forward network with residual connections.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, embed_dim)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out  # Residual connection after attention

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)  # Residual connection after MLP
        return x


class TransformerModel(nn.Module):
    """
    Revised TransformerModel for image upscaling.

    Architecture overview:
      1. A shallow CNN encoder extracts features from the LR image.
      2. A downsampling layer reduces spatial dimensions.
      3. A patch embedding converts the CNN feature map into a sequence of tokens.
         (Using a larger patch size to reduce the total number of tokens.)
      4. A series of transformer blocks processes these tokens.
      5. A patch unembedding converts the tokens back to a spatial feature map.
      6. A CNN decoder refines the features and predicts a residual image.
      7. The predicted residual is added to a bicubic upsampled version of the input,
         producing the final upscaled output.
    """

    def __init__(
            self,
            in_channels=3,
            base_channels=64,
            embed_dim=64,
            transformer_dim=128,
            num_transformer_blocks=8,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1
    ):
        super(TransformerModel, self).__init__()

        # Shallow CNN encoder to extract initial features from the low resolution input
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsample features: from (720,1280) to (360,640)
        self.downsample = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        # Patch Embedding: convert the downsampled feature map to tokens.
        # Using kernel_size=8 and stride=8 on a (360,640) feature map yields tokens of size (45,80)
        # resulting in 45*80 = 3600 tokens instead of 14400.
        self.patch_embed = nn.Conv2d(base_channels, transformer_dim, kernel_size=8, stride=8)
        self.token_H = 360 // 8  # 45
        self.token_W = 640 // 8  # 80
        self.num_tokens = self.token_H * self.token_W  # 3600 tokens

        # Learnable positional embedding for the token sequence.
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, transformer_dim))

        # Transformer blocks for global context
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # Patch Unembedding: convert token sequence back to spatial feature map.
        self.patch_unembed = nn.ConvTranspose2d(transformer_dim, base_channels, kernel_size=8, stride=8)

        # CNN decoder: refine features and predict a residual image.
        self.decoder_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.decoder_conv2 = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass:
          - x: Low resolution image of shape (B, 3, 720, 1280)
          - Returns: Upscaled image of shape (B, 3, 1080, 1920)
        """
        # Compute a global residual: bicubic upscale of the input image.
        upscaled_input = F.interpolate(x, size=(1080, 1920), mode='bicubic', align_corners=False)

        # Encoder: extract features using a shallow CNN.
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))

        # Downsample feature map to reduce spatial dimensions.
        feat_down = self.downsample(feat)  # Shape: (B, base_channels, 360, 640)

        # Patch Embedding: convert feature map to a sequence of tokens.
        tokens = self.patch_embed(feat_down)  # (B, transformer_dim, 45, 80)
        B, C, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, num_tokens, transformer_dim)

        # Add positional embeddings.
        tokens = tokens + self.pos_embed

        # Process tokens through a series of transformer blocks.
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Reshape tokens back into spatial feature maps.
        tokens = tokens.transpose(1, 2).view(B, C, H, W)  # (B, transformer_dim, 45, 80)

        # Patch Unembedding: convert tokens back to CNN feature space.
        feat_trans = self.patch_unembed(tokens)  # (B, base_channels, 360, 640)

        # Combine transformer features with the original downsampled features via skip connection.
        combined_feat = feat_down + feat_trans

        # CNN Decoder: refine features and predict the residual image.
        dec = self.relu(self.decoder_conv1(combined_feat))
        residual = self.decoder_conv2(dec)  # Predicted residual image at lower resolution

        # Upsample the residual to the final output resolution.
        residual_up = F.interpolate(residual, size=(1080, 1920), mode='bicubic', align_corners=False)

        # Add the residual to the bicubic-upscaled input.
        out = upscaled_input + residual_up
        out = torch.clamp(out, 0.0, 1.0)
        return out


# Quick test when running this file directly.
if __name__ == "__main__":
    model = TransformerModel()
    dummy_input = torch.randn(1, 3, 720, 1280)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 3, 1080, 1920)
