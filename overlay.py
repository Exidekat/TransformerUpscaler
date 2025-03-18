"""
overlay.py

This script implements an overlay application for our Transformer upscaler.
It continuously captures a region of the screen (e.g., an application window)
using mss, preprocesses the captured frame to a 720×1280 input, feeds it through
the TransformerModel, and displays the upscaled output (1080×1920) in an OpenCV window.
The upscaled display is updated live, and the window remains on top.
Press "q" to exit the app.

Usage:
    python overlay.py --top 100 --left 100 --width 1280 --height 720 --checkpoint_dir ./checkpoints
"""

import cv2
import mss
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import argparse

from model.TransformerModel import TransformerModel
from tools.utils import get_latest_checkpoint


def main(args):
    # Select device: mps if available, else cuda, otherwise cpu.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the TransformerModel and the latest checkpoint.
    model = TransformerModel().to(device)
    checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Define transform for converting captured image to model input (720x1280, normalized tensor)
    lr_transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    # Set up mss for screen capture of the specified region.
    sct = mss.mss()
    monitor = {"top": args.top, "left": args.left, "width": args.width, "height": args.height}
    print(f"Capturing region: {monitor}")

    # Optionally, set up an OpenCV window that is always on top.
    window_name = "Upscaled Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        frame_start = time.time()

        # Capture the screen region.
        sct_img = sct.grab(monitor)
        # Convert mss image (BGRA) to a NumPy array.
        img = np.array(sct_img)
        # Convert BGRA to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # Create a PIL image.
        pil_img = Image.fromarray(img)

        # Preprocess: resize to 720x1280 and convert to tensor.
        lr_img = lr_transform(pil_img)  # Shape: (3, 720, 1280)
        lr_img = lr_img.unsqueeze(0).to(device)  # Add batch dimension

        # Run inference on the captured frame.
        with torch.no_grad():
            upscaled = model(lr_img)  # Expected output shape: (1, 3, 1080, 1920)

        # Postprocess: remove batch dimension and convert to a PIL image.
        upscaled = upscaled.squeeze(0).cpu()
        upscaled_pil = to_pil(upscaled)
        # Convert to a NumPy array and then to BGR for OpenCV.
        upscaled_np = np.array(upscaled_pil)
        upscaled_np = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)

        # Calculate and display FPS.
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)
        cv2.putText(upscaled_np, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Display the upscaled output.
        cv2.imshow(window_name, upscaled_np)

        # Break if the user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Overlay App for Transformer Upscaler")
    parser.add_argument("--top", type=int, default=100, help="Top coordinate of region to capture")
    parser.add_argument("--left", type=int, default=100, help="Left coordinate of region to capture")
    parser.add_argument("--width", type=int, default=1280, help="Width of region to capture")
    parser.add_argument("--height", type=int, default=720, help="Height of region to capture")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints")
    args = parser.parse_args()
    main(args)
