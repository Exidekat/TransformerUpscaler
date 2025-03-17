"""
data_handling/data_class.py

This module contains the highres_img_dataset, a PyTorch Dataset class
for loading training images for our Transformer upscaler.
It converts each image into a low resolution input (720×1280) and a high resolution
target (1080×1920) so that the network (which outputs Full HD images) can be trained properly.
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

    def __len__(self):
        return len(self.image_files)

    # Returns a tuple: (low resolution input, high resolution target) as RGB image tensors.
    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Open the image and convert it to RGB
        hr_image = Image.open(img_path).convert('RGB')

        # Create a low resolution version for model input (720p)
        lr_transform = transform.Compose([
            transform.Resize((720, 1280)),
            transform.ToTensor()
        ])

        # Create a high resolution version for model target (Full HD: 1080x1920)
        hr_transform = transform.Compose([
            transform.Resize((1080, 1920)),
            transform.ToTensor()
        ])

        lr_image_tensor = lr_transform(hr_image)
        hr_image_tensor = hr_transform(hr_image)

        # Ensure that image tensors are normalized to the [0, 1] range
        assert torch.min(lr_image_tensor) >= 0.0 and torch.max(
            lr_image_tensor) <= 1.0, "LR image tensor not in range [0, 1]"
        assert torch.min(hr_image_tensor) >= 0.0 and torch.max(
            hr_image_tensor) <= 1.0, "HR image tensor not in range [0, 1]"

        return lr_image_tensor, hr_image_tensor
