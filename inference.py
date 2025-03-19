#!/usr/bin/env python
"""
inference.py

This script loads the latest model checkpoint from the specified checkpoint directory,
loads an input image, and performs upscaling using the specified TransformerModel.
The input image is first resized to the desired input resolution (specified by --res_in_h and --res_in_w),
and then the model produces a high resolution output (e.g., 1080×1920) as specified by --res_out.
Mixed precision inference is enabled on CUDA devices via torch.autocast.
The resulting upscaled image is saved to disk.

Usage:
    python inference.py --image_path images/training_set/image_0.jpg --model EfficientTransformer --res_in_h 720 --res_in_w 1280 --res_out 1080 [--compile]
"""

import argparse
import importlib
import torch
from PIL import Image
import torchvision.transforms as transforms
from tools.utils import get_latest_checkpoint, resolutions


def main(args):
    if args.res_out not in resolutions.keys():
        print(f"Resolution {args.res_out} not found in supported output resolutions.")
        exit(-1)
    if args.res_in:
        if args.res_in not in resolutions.keys():
            print(f"Resolution {args.res_in} not found in supported input resolutions.")
            exit(-1)
        res_in = resolutions[args.res_in]  # dynamic input resolution
    else:
        res_in = None

    res_out = resolutions[args.res_out]  # e.g., (1080, 1920)

    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running inference on device: {device}")

    # Dynamically import the desired model module from models/{args.model}/model.py
    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    # Set default checkpoint directory if not provided.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"models/{args.model}/checkpoints"

    # Define transforms.
    lr_transform = transforms.Compose([
        transforms.Resize(res_in),
        transforms.ToTensor()
    ]) if res_in is not None else transforms.Compose([
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    # Load input image and convert to RGB.
    image = Image.open(args.image_path).convert('RGB')
    lr_tensor = lr_transform(image)
    # Optionally, save the downscaled input for inspection.
    downscaled_image = to_pil(lr_tensor)
    downscaled_image.save(args.inp)
    print(f"Downscaled image saved to: {args.inp}")
    lr_tensor = lr_tensor.unsqueeze(0)  # add batch dimension

    # Instantiate and optionally compile the model.
    model = TransformerModel().to(device)

    # Load checkpoint.
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile!")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Run inference under mixed precision).
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        with torch.no_grad():
            output = model(lr_tensor.to(device), res_out)
    output = output.squeeze(0).cpu()
    upscaled_image = to_pil(output)
    upscaled_image.save(args.out)
    print(f"Upscaled image saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Transformer upscaler with dynamic input resolution, Mixed Precision, and Model Selection")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--model", type=str, default="ResidualTransformer",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--res_out", type=str, default='1080',
                        help="Output resolution key (e.g., '1080', '1440', '2160', etc.)")
    parser.add_argument("--res_in", type=str, default=None, help="Input resolution key (None for no downscaling)")
    parser.add_argument("--inp", type=str, default="input.png", help="Output file path for the downscaled input image")
    parser.add_argument("--out", type=str, default="output.png", help="Output file path for the upscaled output image")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation with torch.compile")
    args = parser.parse_args()
    main(args)
