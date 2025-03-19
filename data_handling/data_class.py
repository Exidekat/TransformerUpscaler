"""
data_handling/data_class.py

This module contains the highres_img_dataset, a PyTorch Dataset class
for loading training images for our Transformer upscaler.

For every high-resolution image provided, this dataset generates three unique LR–HR pairs:
  1. LR: 720×1280 → HR: 1080×1920,
  2. LR: 720×1280 → HR: 1440×2560, and
  3. LR: 1080×1920 → HR: 1440×2560.

This augmentation maximizes the use of each image.
"""

import os
from PIL import Image
import torchvision.transforms as transform
from torch.utils.data import Dataset
import torch

class highres_img_dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Collect only .jpg files in the specified directory
        self.image_files = [
            os.path.join(image_dir, file)
            for file in os.listdir(image_dir)
            if file.lower().endswith('.jpg')
        ]
        # Predefined scale pairs for augmentation.
        # Each pair contains 'lr' and 'hr' resolution tuples.
        self.scale_pairs = [
            {"lr": (720, 1280), "hr": (1080, 1920)},
            {"lr": (720, 1280), "hr": (1440, 2560)},
            {"lr": (1080, 1920), "hr": (1440, 2560)},
            {"lr": (720, 1280), "hr": (2160, 3840)},
            {"lr": (1080, 1920), "hr": (2160, 3840)},
            {"lr": (1440, 2560), "hr": (2160, 3840)}
        ]

    def __len__(self):
        # Each image is used to generate all three pairs.
        return len(self.image_files) * len(self.scale_pairs)

    def __getitem__(self, idx):
        # Determine which image and which scale pair to use.
        num_pairs = len(self.scale_pairs)
        image_idx = idx // num_pairs
        pair_idx = idx % num_pairs

        img_path = self.image_files[image_idx]

        # Open the image and convert it to RGB.
        hr_image = Image.open(img_path).convert('RGB')

        # Use the corresponding scale pair.
        pair = self.scale_pairs[pair_idx]

        # Create transforms for LR and HR images.
        lr_transform = transform.Compose([
            transform.Resize(pair["lr"]),
            transform.ToTensor()
        ])
        hr_transform = transform.Compose([
            transform.Resize(pair["hr"]),
            transform.ToTensor()
        ])

        lr_image_tensor = lr_transform(hr_image)
        hr_image_tensor = hr_transform(hr_image)

        # Ensure that image tensors are normalized between 0 and 1.
        assert torch.min(lr_image_tensor) >= 0.0 and torch.max(lr_image_tensor) <= 1.0, "LR image tensor not in range [0, 1]"
        assert torch.min(hr_image_tensor) >= 0.0 and torch.max(hr_image_tensor) <= 1.0, "HR image tensor not in range [0, 1]"

        return lr_image_tensor, hr_image_tensor
