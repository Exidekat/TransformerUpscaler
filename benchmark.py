import os
import argparse
import importlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2ycbcr
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.transform import resize
from tools.utils import get_latest_checkpoint
import time
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

def to_y_channel(img: Image.Image) -> np.ndarray:
    """
    Convert RGB PIL image to Y channel only
    """
    img = np.array(img)
    ycbcr = rgb2ycbcr(img)
    y = ycbcr[:, :, 0]
    return y

def preprocess_image(image: Image.Image, res_in):
    """
    Resize image and convert to tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(res_in, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    return transform(image)

def evaluate_image(model, img_path, scale, device, crop_border) -> tuple:
    # Load original image
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size
    res_in = (orig_h // scale, orig_w // scale)

    # Prepare LR input
    lr_tensor = preprocess_image(image, res_in).unsqueeze(0).to(device)

    # Model Upscale
    with torch.no_grad():
        start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
            sr_tensor = model(lr_tensor, upscale_factor=scale)
        end = time.time()

    inference_time = end - start
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu().clamp(0, 1))

    # Convert to Y channel
    orig_y = to_y_channel(image)
    model_y = to_y_channel(sr_img)

    # Resize if needed
    if model_y.shape != orig_y.shape:
        model_y = resize(model_y, orig_y.shape, anti_aliasing=True)

    # Crop borders
    if crop_border > 0:
        orig_y = orig_y[crop_border:-crop_border, crop_border:-crop_border]
        model_y = model_y[crop_border:-crop_border, crop_border:-crop_border]

    # Compute metrics
    model_ssim = ssim(
        orig_y,
        model_y,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )
    model_psnr = psnr(orig_y, model_y, data_range=255)

    return model_ssim, model_psnr, inference_time

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel
    model = TransformerModel().to(device).eval()

    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir or f"models/{args.model}/checkpoints")
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])

    if args.compile:
        model = torch.compile(model)

    if args.quantize:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    image_paths = [os.path.join(dp, f)
                   for dp, _, filenames in os.walk(args.data_dir)
                   for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    total_ssim = 0
    total_psnr = 0
    total_time = 0

    print(f"\nFound {len(image_paths)} images. Evaluating...\n")

    for img_path in image_paths:
        border = args.scale if args.crop_border else 0
        ssim_val, psnr_val, inf_time = evaluate_image(model, img_path, args.scale, device, border)
        total_ssim += ssim_val
        total_psnr += psnr_val
        total_time += inf_time

        if psnr_val < 26:
            print(f"Low PSNR ({psnr_val:.2f} dB): {img_path}")

    n = len(image_paths)
    print("\n===== AVERAGE RESULTS =====")
    print(f"Model SSIM: {total_ssim/n:.4f}")
    print(f"Model PSNR: {total_psnr/n:.2f} dB")
    print(f"Avg Inference Time: {total_time/n:.4f} seconds")
    print("=============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SSIM/PSNR evaluator for SR models")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="SwinBased", 
                        help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Model checkpoint dir")
    parser.add_argument("--scale", type=int, default=4, 
                        help="Upscale factor (2, 3, 4, 6)")
    parser.add_argument("--crop_border", default=True, 
                        help="Crop border size for metric calculation")
    parser.add_argument("--compile", action="store_true", 
                        help="Enable torch.compile()")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable dynamic quantization")
    args = parser.parse_args()
    main(args)