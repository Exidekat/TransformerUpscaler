#!/usr/bin/env python3
"""
eval_metrics.py

Run a super-resolution model over a dataset and report PSNR/SSIM on the Y (luminance) channel.

Protocol (the crop/border/SSIM conventions used by EDSR and SwinIR):
  1. Crop each HR image to a multiple of `scale` BEFORE downsampling, so the model
     output lands on exactly the HR size and the prediction is never resampled.
  2. Bicubic-downsample the cropped HR to produce the LR input.
  3. Crop `scale` pixels from every border before scoring.
  4. Compute on the Y channel of YCbCr, in fp32, with MATLAB-convention SSIM
     (gaussian_weights=True, sigma=1.5, use_sample_covariance=False).

Use --fast for fp16 autocast when timing throughput; never for published metrics.

NOTE ON COMPARABILITY: the LR input is produced with PIL's BICUBIC resampler, not
MATLAB's `imresize` with antialiasing, which is what the SR literature uses. That
difference shifts the whole scale by roughly a decibel, so numbers from this harness
are internally comparable to each other -- including the bicubic reference run through
the same pipeline -- but should not be placed alongside published benchmark tables.
"""
import os
import argparse
import importlib
import time
import warnings

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from tools.utils import get_latest_checkpoint


def main(args):
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Device selection: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    import_safe_model_arg = str(args.model).replace("/", '.')
    model_module = importlib.import_module(f"models.{import_safe_model_arg}.model")
    TransformerModel = getattr(model_module, "TransformerModel")
    model = TransformerModel().to(device)
    # Load latest checkpoint
    ckpt_dir = args.checkpoint_dir or f"models/{args.model}/checkpoints"
    ckpt_path, _ = get_latest_checkpoint(ckpt_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Support checkpoints saved as {'model_state_dict': ...} or raw state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    # Optional compilation
    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    # Optional quantization
    if args.quantize:
        import torch.nn as nn
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        print("Model quantized dynamically")

    # Prepare transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Collect image files
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = sorted(
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.lower().endswith(exts)
    )
    if not image_files:
        print(f"No images found in directory: {args.data_dir}")
        return

    psnr_vals = []
    ssim_vals = []
    times = []
    total = len(image_files)
    print(f"Evaluating {total} images at scale {args.scale}...")

    for idx, img_path in enumerate(image_files):
        if args.max_images and idx >= args.max_images:
            break
        # Load high-resolution image
        hr_img = Image.open(img_path).convert('RGB')
        w, h = hr_img.size
        # Crop HR to a multiple of the scale factor so SR output matches HR exactly
        w, h = w - (w % args.scale), h - (h % args.scale)
        if (w, h) != hr_img.size:
            hr_img = hr_img.crop((0, 0, w, h))
        # Create low-resolution input by bicubic downsampling the cropped HR
        lr_img = hr_img.resize((w // args.scale, h // args.scale), Image.BICUBIC)

        # Model inference
        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            # fp32 by default for published metrics; --fast enables fp16 autocast for timing
            if args.fast and device.type in ('cuda', 'mps'):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    start = time.time()
                    out = model(lr_tensor, upscale_factor=args.scale)
                    end = time.time()
            else:
                start = time.time()
                out = model(lr_tensor, upscale_factor=args.scale)
                end = time.time()
        times.append(end - start)

        # Convert model output to PIL image. A size mismatch is a bug in the upsampler,
        # not something to paper over by resampling the prediction.
        pred_img = to_pil(out.float().squeeze(0).cpu().clamp(0, 1))
        if pred_img.size != hr_img.size:
            raise RuntimeError(
                f"{os.path.basename(img_path)}: model returned {pred_img.size}, expected "
                f"{hr_img.size} at scale {args.scale}. Resampling the prediction would "
                f"invalidate the metric, so this is a hard error."
            )

        # Convert to Y channel
        hr_y = np.array(hr_img.convert('YCbCr').split()[0], dtype=np.float32) / 255.0
        pred_y = np.array(pred_img.convert('YCbCr').split()[0], dtype=np.float32) / 255.0

        # Standard protocol: ignore `scale` pixels at each border
        b = args.scale
        if b > 0 and hr_y.shape[0] > 2 * b and hr_y.shape[1] > 2 * b:
            hr_y, pred_y = hr_y[b:-b, b:-b], pred_y[b:-b, b:-b]

        # Compute metrics (MATLAB-convention SSIM, as used in the SR literature)
        p = compare_psnr(hr_y, pred_y, data_range=1.0)
        s = compare_ssim(hr_y, pred_y, data_range=1.0,
                         gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False)
        psnr_vals.append(p)
        ssim_vals.append(s)

        # Logging
        if (idx + 1) % args.log_interval == 0:
            avg_p = sum(psnr_vals) / len(psnr_vals)
            avg_s = sum(ssim_vals) / len(ssim_vals)
            avg_t = sum(times) / len(times)
            print(f"[{idx+1}/{total}] Avg PSNR: {avg_p:.3f} dB, Avg SSIM: {avg_s:.4f}, Avg time: {avg_t:.4f}s")

    # Final results
    count = len(psnr_vals)
    avg_psnr = sum(psnr_vals) / count if count else 0
    avg_ssim = sum(ssim_vals) / count if count else 0
    avg_time = sum(times) / count if count else 0
    summary = (
        f"===== Evaluation Complete =====\n"
        f"Images evaluated: {count}\n"
        f"Average PSNR (Y): {avg_psnr:.3f} dB\n"
        f"Average SSIM (Y): {avg_ssim:.4f}\n"
        f"Average inference time: {avg_time:.4f} seconds\n"
        f"==============================="
    )
    print(summary)
    # Save average PSNR and SSIM to a file with detailed naming in the model directory
    from datetime import datetime
    # Date in DDMMYY format
    date_str = datetime.now().strftime("%d%m%y")
    # Build filename components
    scale_str = f"scale{args.scale}"
    # Sanitize data_dir for filename (replace slashes)
    datadir_str = os.path.basename(args.data_dir.rstrip('/')) or "dataset"
    flags = ""
    if args.compile:
        flags += "_compile"
    if args.quantize:
        flags += "_quantize"
    # Determine model directory (e.g., models/Fastv2)
    model_dir = os.path.join("models", args.model)
    os.makedirs(model_dir, exist_ok=True)
    score_filename = f"score_{date_str}_{scale_str}_{datadir_str}{flags}.txt"
    score_path = os.path.join(model_dir, score_filename)
    try:
        with open(score_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {datadir_str}   Scale: x{args.scale}   Images: {count}\n")
            f.write(f"Protocol: HR cropped to multiple of scale; bicubic down; "
                    f"{args.scale}px border crop; Y channel; "
                    f"{'fp16' if args.fast else 'fp32'}; MATLAB-convention SSIM\n")
            f.write(f"Average PSNR (Y): {avg_psnr:.3f} dB\n")
            f.write(f"Average SSIM (Y): {avg_ssim:.4f}\n")
        print(f"Scores saved to: {score_path}")
    except Exception as e:
        print(f"Failed to save scores to file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate super-resolution model on full dataset and compute PSNR/SSIM on Y channel'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing high-resolution images')
    parser.add_argument('--model', type=str, default='Fastv2',
                        help='Model name (folder under models/)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory for model')
    parser.add_argument('--scale', type=int, required=True,
                        help='Upscale factor (e.g., 2, 3, 4, 6)')
    parser.add_argument('--max_images', type=int, default=100,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log progress every N images')
    parser.add_argument('--fast', action='store_true',
                        help='fp16 autocast on CUDA/MPS. For timing runs only, never for published metrics')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for model')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable dynamic quantization of model')
    args = parser.parse_args()
    main(args)