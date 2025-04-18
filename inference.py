#!/usr/bin/env python
"""
inference_quantized.py

This script loads the latest model checkpoint from the specified checkpoint directory,
loads an input image, and performs upscaling using the specified TransformerModel.
It supports model quantization to reduce the footprint and improve inference speed.
The input image is first resized to the desired input resolution (specified by --res_in),
or, if --downscale is set, it is automatically downscaled by the scale factor.
Then the model produces a high resolution output as specified by --scale.
Quantization is applied post-training dynamically to all nn.Linear layers.
Mixed precision inference is enabled on CUDA/MPS devices via torch.autocast.
The resulting upscaled image is saved to disk.

Usage:
    python inference.py --image_path images/training_set/image_0.jpg --model StrippedTransformer --res_in 720 --scale 3 [--compile] [--quantize] [--downscale]
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
import argparse
import importlib
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import img_as_float, io   
from skimage.transform import resize
import torchvision.transforms as transforms
from tools.utils import get_latest_checkpoint, resolutions
import torch.nn as nn
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):
    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running inference on device: {device}")

    # Validate scale factor.
    if args.scale not in [2, 3, 4, 6]:
        print(f"Scale factor {args.scale} not supported.")
        exit(-1)
    
    # Determine input resolution.
    # If --downscale is set, ignore --res_in and compute resolution from the original image.
    if args.downscale:
        # Open image temporarily to get its dimensions.
        temp_image = Image.open(args.image_path).convert('RGB')
        orig_w, orig_h = temp_image.size  # PIL returns (width, height)
        # Downscale input resolution: each dimension divided by the scale factor.
        res_in = (orig_h // args.scale, orig_w // args.scale)
        print(f"Downscale flag set: using computed downscaled resolution {res_in}")
    elif args.res_in:
        if args.res_in not in resolutions.keys():
            print(f"Resolution {args.res_in} not found in supported input resolutions.")
            exit(-1)
        res_in = resolutions[args.res_in]  # e.g. resolutions might be a dict mapping "720" -> (720, 1280) etc.
    else:
        res_in = None

    # Define transforms based on res_in.
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
    # Save the downscaled input for inspection.
    downscaled_image = to_pil(lr_tensor)
    downscaled_image.save(args.inp)
    print(f"Downscaled image saved to: {args.inp}")
    
    # Bicubic interpolation baseline.
    bicubic_image = to_pil(lr_tensor)
    bicubic_image = bicubic_image.resize(
        (lr_tensor.shape[2] * args.scale, lr_tensor.shape[1] * args.scale), Image.BICUBIC
    )
    bicubic_image.save('bicubic.jpg')
    print(f"Bicubic image saved to: bicubic.jpg")
    
    lr_tensor = lr_tensor.unsqueeze(0)  # Add batch dimension.

    # Instantiate the model.
    import_safe_model_arg = str(args.model).replace("/", '.')
    model_module = importlib.import_module(f"models.{import_safe_model_arg}.model")
    TransformerModel = model_module.TransformerModel
    model = TransformerModel().to(device)

    # Load checkpoint.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"models/{args.model}/checkpoints"
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Support checkpoints saved as {'model_state_dict': ...} or raw state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()

    # Optionally compile the model.
    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile!")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Optionally apply dynamic quantization.
    if args.quantize:
        print("Applying dynamic quantization to the model...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        print("Model quantization complete.")

    # Run inference with mixed precision if possible.
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu"))
    inf_time = 0
    with torch.no_grad():
        if device.type in ['cuda', 'mps']:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                start = time.time()
                output = model(lr_tensor.to(device), upscale_factor=args.scale)
                end = time.time()
                inf_time = end - start
        else:
            start = time.time()
            output = model(lr_tensor.to(device), upscale_factor=args.scale)
            end = time.time()
            inf_time = end - start
    output = output.squeeze(0).cpu()
    upscaled_image = to_pil(output)
    upscaled_image.save(args.out)
    print(f"Upscaled image saved to: {args.out}")
    
    # Calculate SSIM and PSNR.
    original = img_as_float(io.imread(args.image_path))
    pred = img_as_float(io.imread(args.out))
    if original.shape[0] != pred.shape[0] or original.shape[1] != pred.shape[1]:
        original = resize(original, (pred.shape[0], pred.shape[1]))
    lowres = img_as_float(io.imread(args.inp))
    lowres = resize(lowres, (original.shape[0], original.shape[1]))
    
    model_ssim_val = ssim(original, pred, data_range=1, channel_axis=-1)
    model_psnr_val = psnr(original, pred, data_range=1)
    
    bicubic_ssim_val = ssim(original, lowres, data_range=1, channel_axis=-1)
    bicubic_psnr_val = psnr(original, lowres, data_range=1)
    
    print('====================')
    print(f"Bicubic Scores:\tSSIM: {bicubic_ssim_val:.4f}, PSNR: {bicubic_psnr_val:.2f} dB")
    print(f"Model Scores:\tSSIM: {model_ssim_val:.4f}, PSNR: {model_psnr_val:.2f} dB")
    print(f"Model has {n_params} trainable parameters, Inference time was {inf_time:.4f} seconds")
    print('====================')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Transformer upscaler with dynamic input resolution, quantization, and optional downscaling"
    )
    parser.add_argument("--image_path", type=str, default="images/training_set/image_100.jpg",
                        help="Path to the input image file")
    parser.add_argument("--model", type=str, default="SwinBased",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--scale", type=int, default=3,
                        help="Output upscale factor (e.g. 2, 3, 4, 6)")
    parser.add_argument("--res_in", type=str, default=None,
                        help="Input resolution key (ignored if --downscale is set)")
    parser.add_argument("--downscale", action="store_true",
                        help="If set, downscale the input image with bicubic (using original dimensions divided by scale) "
                             "and override --res_in.")
    parser.add_argument("--inp", type=str, default="input.jpg",
                        help="Output file path for the downscaled input image")
    parser.add_argument("--out", type=str, default="model.jpg",
                        help="Output file path for the upscaled output image")
    parser.add_argument("--compile", action="store_true",
                        help="Enable model compilation with torch.compile")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable dynamic quantization on the model to reduce footprint")
    args = parser.parse_args()
    main(args)
