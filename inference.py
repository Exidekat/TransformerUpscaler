"""
inference.py

This script loads the latest model checkpoint, loads an input image, and performs upscaling using the TransformerModel.
The input image is first resized to the desired input resolution (specified by --res_in_h and --res_in_w),
and then the model produces a high resolution output (e.g., 1080×1920) as specified by --res_out.
The resulting upscaled image is saved to disk.

Usage:
    python inference.py --image_path images/training_set/image_0.jpg --res_in_h 720 --res_in_w 1280 --res_out 1080
"""

import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.TransformerModel import TransformerModel
from tools.utils import get_latest_checkpoint, resolutions

def main(args):
    if args.res_out not in resolutions.keys():
        print(f"Resolution {args.res_out} not found in supported output resolutions.")
        exit(-1)
    res_out = resolutions[args.res_out]  # e.g., (1080, 1920)
    res_in = (args.res_in_h, args.res_in_w)  # dynamic input resolution

    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running inference on device: {device}")

    # Define transforms.
    lr_transform = transforms.Compose([
        transforms.Resize(res_in),
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

    # Load model.
    model = TransformerModel().to(device)
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference.
    with torch.no_grad():
        output = model(lr_tensor.to(device), res_out)
    output = output.squeeze(0).cpu()
    upscaled_image = to_pil(output)
    upscaled_image.save(args.out)
    print(f"Upscaled image saved to: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Transformer upscaler with dynamic input resolution")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--res_out", type=str, default='1080',
                        help="Output resolution key (e.g., '1080', '1440', '2160', etc.)")
    parser.add_argument("--res_in_h", type=int, default=720,
                        help="Input resolution height (downscaled for model)")
    parser.add_argument("--res_in_w", type=int, default=1280,
                        help="Input resolution width (downscaled for model)")
    parser.add_argument("--inp", type=str, default="input.png",
                        help="Output file path for the downscaled input image")
    parser.add_argument("--out", type=str, default="output.png",
                        help="Output file path for the upscaled output image")
    args = parser.parse_args()
    main(args)
