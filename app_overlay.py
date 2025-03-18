#!/usr/bin/env python
"""
app_overlay.py

This script provides an interactive window selection and overlay for the Transformer upscaler.
It works cross‑platform:
  - On macOS, it uses Quartz (via PyObjC) to list windows and capture the content of the selected window.
  - On Windows, it uses pygetwindow to list windows and PIL.ImageGrab to capture the window content.
  - On Linux, it falls back to capturing a screen region with mss (note: this may capture the overlay).

Once the user selects a window, the script continuously captures the _target window’s_ content (using OS‑specific methods),
runs our TransformerModel on the captured image (after resizing it to 720×1280),
and displays an overlay (via OpenCV) that is resized to cover the target window.
The overlay window is repositioned so it always sits exactly over the target.
Press "q" to exit.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import argparse
import platform
import mss

# Platform‐specific imports
if platform.system() == "Darwin":
    import Quartz
elif platform.system() == "Windows":
    import pygetwindow as gw
else:
    # On Linux we still try to use pygetwindow for listing window titles.
    import pygetwindow as gw

from model.TransformerModel import TransformerModel
from tools.utils import get_latest_checkpoint


# ========= macOS FUNCTIONS =========
def list_windows_macos():
    """
    Retrieve a list of on-screen windows (as dictionaries) with non‑empty titles using Quartz.
    """
    window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    windows = []
    for window in window_info_list:
        title = window.get('kCGWindowName', '')
        if title and title.strip() != "":
            windows.append(window)
    return windows


def select_window_macos():
    """List available macOS windows and let the user select one by number."""
    windows = list_windows_macos()
    if not windows:
        raise Exception("No windows found on macOS.")
    print("Available windows:")
    for i, window in enumerate(windows, start=1):
        title = window.get('kCGWindowName', 'No Title')
        print(f"{i}: {title}")
    idx = int(input("Enter the number of the window to capture: "))
    return windows[idx - 1]


def get_window_bounds_macos(window):
    """
    Given a Quartz window dictionary, extract its bounding box as (left, top, width, height).
    Note: Quartz coordinates have origin at the bottom-left.
    """
    bounds = window.get('kCGWindowBounds', {})
    left = int(bounds.get('X', 0))
    top = int(bounds.get('Y', 0))
    width = int(bounds.get('Width', 0))
    height = int(bounds.get('Height', 0))
    return left, top, width, height


def capture_window_content_macos(window):
    """
    Capture the content of a macOS window using Quartz’s CGWindowListCreateImage.
    Returns a PIL Image in RGB.
    """
    from Quartz import CGWindowListCreateImage, kCGWindowListOptionIncludingWindow, kCGWindowImageDefault, CGRectMake
    window_id = window.get('kCGWindowNumber')
    bounds = window.get('kCGWindowBounds', {})
    x = float(bounds.get('X', 0))
    y = float(bounds.get('Y', 0))
    width = float(bounds.get('Width', 0))
    height = float(bounds.get('Height', 0))
    rect = CGRectMake(x, y, width, height)
    cg_image = CGWindowListCreateImage(rect, kCGWindowListOptionIncludingWindow, window_id, kCGWindowImageDefault)
    from PIL import Image
    import Quartz
    if cg_image is None:
        return None
    w = Quartz.CGImageGetWidth(cg_image)
    h = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    data_provider = Quartz.CGImageGetDataProvider(cg_image)
    data = Quartz.CGDataProviderCopyData(data_provider)
    # Create a PIL image from the raw data (assumed to be in RGBA).
    img = Image.frombuffer("RGBA", (w, h), data, "raw", "RGBA", bytes_per_row, 1)
    return img.convert("RGB")


# ========= WINDOWS FUNCTIONS =========
def select_window_windows():
    """List available windows using pygetwindow and let the user select one by number."""
    titles = gw.getAllTitles()
    titles = [title for title in titles if title.strip() != ""]
    if not titles:
        raise Exception("No windows found.")
    print("Available windows:")
    for i, title in enumerate(titles, start=1):
        print(f"{i}: {title}")
    idx = int(input("Enter the number of the window to capture: "))
    selected_title = titles[idx - 1]
    # Try to retrieve the window object.
    if hasattr(gw, "getWindowsWithTitle"):
        windows = gw.getWindowsWithTitle(selected_title)
        if windows:
            return windows[0]
    # Fallback: prompt the user for coordinates.
    print(f"Could not automatically retrieve window object for '{selected_title}'.")
    left = int(input("Left: "))
    top = int(input("Top: "))
    width = int(input("Width: "))
    height = int(input("Height: "))

    # Create a dummy object with these attributes.
    class DummyWindow:
        pass

    win = DummyWindow()
    win.left = left
    win.top = top
    win.width = width
    win.height = height
    return win


def capture_window_content_windows(win):
    """
    Capture the content of a Windows window using PIL.ImageGrab.
    Returns a PIL Image in RGB.
    """
    from PIL import ImageGrab
    bbox = (win.left, win.top, win.left + win.width, win.top + win.height)
    return ImageGrab.grab(bbox).convert("RGB")


# ========= LINUX FUNCTIONS =========
def capture_window_content_linux(monitor):
    """
    Fallback: Capture a region of the screen using mss on Linux.
    Note: This method captures whatever is visible on the screen in the region.
    """
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)


# ========= MAIN APP =========
def main(args):
    sys_platform = platform.system()
    if sys_platform == "Darwin":
        selected_window = select_window_macos()
        selected_title = selected_window.get('kCGWindowName', 'No Title')
        print(f"Selected window: {selected_title}")
        left, top, width, height = get_window_bounds_macos(selected_window)
        # For macOS, we capture via Quartz.
        capture_func = lambda: capture_window_content_macos(selected_window)
    elif sys_platform == "Windows":
        win = select_window_windows()
        print(f"Selected window bounds: left={win.left}, top={win.top}, width={win.width}, height={win.height}")
        capture_func = lambda: capture_window_content_windows(win)
        left, top, width, height = win.left, win.top, win.width, win.height
    else:
        # Linux fallback: use pygetwindow to get bounding box.
        win = select_window_windows()
        print(f"Selected window bounds: left={win.left}, top={win.top}, width={win.width}, height={win.height}")
        monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        capture_func = lambda: capture_window_content_linux(monitor)
        left, top, width, height = win.left, win.top, win.width, win.height

    print(f"Using bounding box: left={left}, top={top}, width={width}, height={height}")

    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the TransformerModel and latest checkpoint.
    model = TransformerModel().to(device)
    checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Define transforms.
    lr_transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    # Create an OpenCV overlay window.
    window_name = "Overlay Upscaled"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        frame_start = time.time()
        # Capture the target window's content using OS-specific API.
        captured_img = capture_func()
        if captured_img is None:
            print("Failed to capture window content.")
            continue

        # Preprocess: resize captured image to 720x1280.
        lr_img = lr_transform(captured_img).unsqueeze(0).to(device)

        # Run inference.
        with torch.no_grad():
            upscaled = model(lr_img)  # Expected output: (1, 3, 1080, 1920)
        upscaled = upscaled.squeeze(0).cpu()
        upscaled_pil = to_pil(upscaled)
        upscaled_np = np.array(upscaled_pil)
        upscaled_np = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)

        # Resize upscaled image to match the target window dimensions.
        overlay = cv2.resize(upscaled_np, (width, height))

        # Reposition the overlay window to match the target window.
        cv2.moveWindow(window_name, left, top)

        # Calculate and display FPS.
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)
        cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow(window_name, overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay App for Transformer Upscaler with OS-Specific Window Capture"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints")
    args = parser.parse_args()
    main(args)
