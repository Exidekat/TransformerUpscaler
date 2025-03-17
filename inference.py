"""
inference.py

This script loads the latest model checkpoint from the specified directory,
loads an input image, and performs upscaling using the TransformerModel.
The input image is first resized to 720×1280 (low resolution) to match the model's input.
The model produces a Full HD output (1080×1920), which is then saved to the specified output file.
"""

import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.TransformerModel import TransformerModel
from tools.utils import get_latest_checkpoint

resolutions = {
    '1080': (1080, 1920),
    '1440': (1440, 2560),
    '2k': (1440, 2560),
    '2160': (2160, 3840),
    '4k': (2160, 3840)
}

def main(args):
    # if no gpu available, use cpu. if on macos>=13.0, use mps
    DEVICE = "cpu"

    if torch.backends.mps.is_built():
        DEVICE = "mps"
    elif torch.backends.cuda.is_built():
        DEVICE = "cuda"

    device = torch.device(DEVICE)
    print(f"Running inference on device: {device}")

    if args.res_out not in resolutions.keys():
        print(f"Resolution {args.res_out} not found in supported output resolutions.")
        exit(-1)
    res = resolutions[args.res_out]

    # For saving images with Pillow
    to_pil = transforms.ToPILImage()

    # Load input image and apply low resolution transform (720x1280)
    image = Image.open(args.image_path).convert('RGB')
    lr_transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
    lr_tensor = lr_transform(image)

    # Saving the 720p input image
    downscaled_image = to_pil(lr_tensor)
    downscaled_image.save(args.inp)
    print(f"Downscaled image saved to: {args.inp}")

    lr_tensor = lr_tensor.unsqueeze(0)  # Add batch dimension

    # Instantiate the model and load the latest checkpoint
    model = TransformerModel().to(device)
    checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference with no gradient tracking
    with torch.no_grad():
        output = model(lr_tensor.to(device), res)

    # Convert model output tensor to PIL image (output shape: (1,3,1080,1920))
    output = output.squeeze(0).cpu()  # Remove batch dimension

    upscaled_image = to_pil(output)

    # Save the upscaled image to the specified output path
    upscaled_image.save(args.out)
    print(f"Upscaled image saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Transformer upscaler")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--res_out", type=str, default='1080',
                        help="Output resolution")
    parser.add_argument("--inp", type=str, default="input.png",
                        help="Output file path for the downscaled input image")
    parser.add_argument("--out", type=str, default="output.png",
                        help="Output file path for the upscaled output image")
    args = parser.parse_args()
    main(args)
