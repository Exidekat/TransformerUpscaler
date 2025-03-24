#!/usr/bin/env python
"""
data_handling/data_class.py

This module contains two Dataset classes for the Transformer upscaler.
1. highres_img_dataset: loads images from a local directory.
2. highres_img_dataset_online: downloads 4K images online asynchronously.
   For every high-resolution image, it generates multiple LR–HR pairs based on predefined scale pairs.
   When the cache falls below a threshold, a background thread concurrently downloads a batch of new images.
"""

import os
import time
import threading
from collections import deque
from io import BytesIO
import requests
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import concurrent.futures

class highres_img_dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Collect only .jpg files in the specified directory.
        self.image_files = [
            os.path.join(image_dir, file)
            for file in os.listdir(image_dir)
            if file.lower().endswith('.jpg')
        ]
        # Predefined scale pairs.
        self.scale_pairs = [
            {"lr": (720, 1280), "hr": (1080, 1920)}
            # {"lr": (720, 1280), "hr": (1440, 2560)},
            # {"lr": (1080, 1920), "hr": (1440, 2560)},
            # {"lr": (720, 1280), "hr": (2160, 3840)},
            # {"lr": (1080, 1920), "hr": (2160, 3840)},
            # {"lr": (1440, 2560), "hr": (2160, 3840)}
        ]

    def __len__(self):
        # Each image is used to generate all scale pairs.
        # return len(self.image_files) * len(self.scale_pairs)
        return 50

    def __getitem__(self, idx):
        num_pairs = len(self.scale_pairs)
        image_idx = idx // num_pairs
        pair_idx = idx % num_pairs

        img_path = self.image_files[image_idx]
        hr_image = Image.open(img_path).convert('RGB')

        pair = self.scale_pairs[pair_idx]
        lr_transform = transforms.Compose([
            transforms.Resize(pair["lr"]),
            transforms.ToTensor()
        ])
        hr_transform = transforms.Compose([
            transforms.Resize(pair["hr"]),
            transforms.ToTensor()
        ])

        lr_image_tensor = lr_transform(hr_image)
        hr_image_tensor = hr_transform(hr_image)

        # Ensure image tensors are in [0,1]
        assert torch.min(lr_image_tensor) >= 0.0 and torch.max(lr_image_tensor) <= 1.0, "LR image tensor not in range [0, 1]"
        assert torch.min(hr_image_tensor) >= 0.0 and torch.max(hr_image_tensor) <= 1.0, "HR image tensor not in range [0, 1]"

        return lr_image_tensor, hr_image_tensor

class highres_img_dataset_online(Dataset):
    """
    A PyTorch Dataset that downloads high-resolution 4K images online asynchronously.

    This dataset maintains a cache (a deque) of downloaded images.
    When __getitem__ is called, the first cached image is used to produce one LR–HR pair (based on a predefined scale pair).
    Each image is used for exactly len(scale_pairs) items before being discarded.
    If the cache size falls below a threshold, a background thread concurrently downloads a batch of new images.
    """
    def __init__(self):
        super().__init__()
        self.cache = deque()  # Each item is a tuple (PIL.Image, used_count)
        # self.scale_pairs = [
        #     {"lr": (720, 1280), "hr": (1080, 1920)},
        #     {"lr": (720, 1280), "hr": (1440, 2560)},
        #     {"lr": (1080, 1920), "hr": (1440, 2560)},
        #     {"lr": (720, 1280), "hr": (2160, 3840)},
        #     {"lr": (1080, 1920), "hr": (2160, 3840)},
        #     {"lr": (1440, 2560), "hr": (2160, 3840)}
        # ]
        
        self.scale_pairs = [
            {"lr": (96, 96), "hr": (192, 192)},
            {"lr": (96, 96), "hr": (288, 288)},
            {"lr": (96, 96), "hr": (384, 384)},
            {"lr": (96, 96), "hr": (576, 576)}
        ]
        self.num_scale_pairs = len(self.scale_pairs)
        self.batch_download_count = 50  # Number of images to download per batch.
        self.minimum_cache = 10         # Minimum images to maintain in cache.
        # The following objects are not picklable, so we will reinitialize them after pickling.
        self.download_lock = threading.Lock()
        self.stop_event = threading.Event()
        # Start background thread to manage the cache.
        self.download_thread = threading.Thread(target=self._download_loop, daemon=True)
        self.download_thread.start()

    def _download_image(self):
        url = "https://picsum.photos/3840/2160"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def _download_batch(self):
        """Download a batch of images concurrently."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._download_image) for _ in range(self.batch_download_count)]
            for future in concurrent.futures.as_completed(futures):
                img = future.result()
                if img is not None:
                    with self.download_lock:
                        self.cache.append((img, 0))

    def _download_loop(self):
        while not self.stop_event.is_set():
            with self.download_lock:
                current_cache = len(self.cache)
            if current_cache < self.minimum_cache:
                self._download_batch()
            else:
                time.sleep(1)

    def __len__(self):
        # Emulate an big online dataset.
        return 500

    def __getitem__(self, idx):
        # Wait until at least one image is available.
        while True:
            with self.download_lock:
                if len(self.cache) > 0:
                    img, used_count = self.cache[0]
                    break
            time.sleep(0.05)
        pair = self.scale_pairs[used_count]
        lr_transform = transforms.Compose([
            transforms.Resize(pair["lr"]),
            transforms.ToTensor()
        ])
        hr_transform = transforms.Compose([
            transforms.Resize(pair["hr"]),
            transforms.ToTensor()
        ])
        lr_image_tensor = lr_transform(img)
        hr_image_tensor = hr_transform(img)
        assert torch.min(lr_image_tensor) >= 0.0 and torch.max(lr_image_tensor) <= 1.0, "LR image tensor not in range [0,1]"
        assert torch.min(hr_image_tensor) >= 0.0 and torch.max(hr_image_tensor) <= 1.0, "HR image tensor not in range [0,1]"
        with self.download_lock:
            new_used_count = used_count + 1
            if new_used_count >= self.num_scale_pairs:
                self.cache.popleft()
            else:
                self.cache[0] = (img, new_used_count)
        return lr_image_tensor, hr_image_tensor

    def __getstate__(self):
        """
        Custom pickling: remove unpicklable objects.
        """
        state = self.__dict__.copy()
        # Remove the lock, event, and thread.
        state.pop("download_lock", None)
        state.pop("stop_event", None)
        state.pop("download_thread", None)
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling.
        """
        self.__dict__.update(state)
        # Reinitialize unpicklable objects.
        self.download_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.download_thread = threading.Thread(target=self._download_loop, daemon=True)
        self.download_thread.start()

    def __del__(self):
        self.stop_event.set()
        if hasattr(self, "download_thread") and self.download_thread.is_alive():
            self.download_thread.join()

# Quick test.
if __name__ == "__main__":
    dataset = highres_img_dataset_online()
    print("Online dataset length (simulated):", len(dataset))
    lr, hr = dataset[0]
    print("LR shape:", lr.shape, "HR shape:", hr.shape)
