#!/usr/bin/env python
"""
app_overlay.py

This script provides an interactive window selection and overlay for the Transformer upscaler.
It works cross‑platform:
  - On macOS, it uses Quartz (via PyObjC) to list windows and capture the content of the selected window.
  - On Windows, it uses pygetwindow to list windows and PIL.ImageGrab to capture the window content.
  - On Linux, it falls back to capturing a screen region using mss.

Once the user selects a window, the script continuously captures the _target window’s_ content
(using OS‑specific methods), runs our TransformerModel on the captured image (after resizing to 720×1280),
and displays an overlay (via OpenCV) that is resized to cover the target window.
Additionally, on macOS the overlay window is adjusted upward slightly, and is set to be click‑through.
Press "q" to exit.
"""
import importlib

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import argparse
import platform
import mss

# Platform‑specific imports
if platform.system() == "Darwin":
    import Quartz
    from AppKit import NSApplication
elif platform.system() == "Windows":
    import pygetwindow as gw
else:
    import pygetwindow as gw  # Linux fallback

from tools.utils import get_latest_checkpoint, resolutions


# ========= macOS FUNCTIONS =========
def list_windows_macos():
    """Retrieve a list of on‑screen windows (as dictionaries) with non‑empty titles using Quartz."""
    window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    windows = []
    for window in window_info_list:
        title = window.get('kCGWindowName', '')
        if title and title.strip() != "":
            windows.append(window)
    return windows


def select_window_macos():
    """List available macOS windows (using Quartz) and let the user select one by number."""
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
    Note: Quartz coordinates have origin at the bottom‑left.
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
    if cg_image is None:
        return None
    # Get image dimensions.
    import Quartz
    w = Quartz.CGImageGetWidth(cg_image)
    h = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    data_provider = Quartz.CGImageGetDataProvider(cg_image)
    data = Quartz.CGDataProviderCopyData(data_provider)
    img = Image.frombuffer("RGBA", (w, h), data, "raw", "RGBA", bytes_per_row, 1)
    return img.convert("RGB")


def set_overlay_passthrough_macos(window_title):
    """
    Find the NSWindow with the given title and set it to ignore mouse events.
    This makes the overlay window click‑through.
    """
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()
    # Give the system a moment to create the window.
    time.sleep(0.5)
    for win in app.windows():
        # The title may be an NSString; convert to Python string.
        if window_title in str(win.title()):
            win.setIgnoresMouseEvents_(True)
            print(f"Set window '{win.title()}' to be click-through.")
            return
    print("Could not set overlay window to click-through.")


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
    if hasattr(gw, "getWindowsWithTitle"):
        windows = gw.getWindowsWithTitle(selected_title)
        if windows:
            return windows[0]
    print(f"Could not automatically retrieve window object for '{selected_title}'.")
    left = int(input("Left: "))
    top = int(input("Top: "))
    width = int(input("Width: "))
    height = int(input("Height: "))
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
        # On macOS, the overlay window is observed to be slightly low;
        # adjust the top coordinate by subtracting an offset.
        top = max(0, top - 65)
        capture_func = lambda: capture_window_content_macos(selected_window)
    elif sys_platform == "Windows":
        win = select_window_windows()
        print(f"Selected window bounds: left={win.left}, top={win.top}, width={win.width}, height={win.height}")
        capture_func = lambda: capture_window_content_windows(win)
        left, top, width, height = win.left, win.top, win.width, win.height
    else:
        # Linux fallback.
        win = select_window_windows()
        print(f"Selected window bounds: left={win.left}, top={win.top}, width={win.width}, height={win.height}")
        monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        capture_func = lambda: capture_window_content_linux(monitor)
        left, top, width, height = win.left, win.top, win.width, win.height

    print(f"Using bounding box: left={left}, top={top}, width={width}, height={height}")

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
    print(f"Using device: {device}")

    # Dynamically import the desired model module from models/{args.model}/model.py
    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    # Set default checkpoint directory if not provided.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"models/{args.model}/checkpoints"

    # Instantiate the model and load the latest checkpoint
    model = TransformerModel().to(device)
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Define transforms.
    lr_transform = transforms.Compose([
        transforms.Resize(res_in),
        transforms.ToTensor()
    ]) if res_in is not None else transforms.Compose([
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    # Create an OpenCV window for the overlay.
    window_name = "Overlay Upscaled"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    # On macOS, attempt to set the overlay window to be click-through.
    if sys_platform == "Darwin":
        # Give a moment for the window to be created.
        time.sleep(0.5)
        set_overlay_passthrough_macos(window_name)

    while True:
        frame_start = time.time()
        captured_img = capture_func()
        if captured_img is None:
            print("Failed to capture window content.")
            continue

        # Preprocess: resize captured image to 720x1280.
        lr_img = lr_transform(captured_img).unsqueeze(0).to(device)

        # Run inference.
        with torch.no_grad():
            upscaled = model(lr_img, res_out)  # Expected output: (1, 3, res[0], res[1])
        upscaled = upscaled.squeeze(0).cpu()
        upscaled_pil = to_pil(upscaled)
        upscaled_np = np.array(upscaled_pil)

        # Resize the output to match the target window dimensions.
        overlay = cv2.resize(upscaled_np, (width, height))

        # Reposition the overlay window.
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
        description="Overlay App for Transformer Upscaler with OS-Specific Window Capture and Passthrough Overlay (Cross-Platform)"
    )
    parser.add_argument("--model", type=str, default="EfficientTransformer",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--res_out", type=str, default='4k',
                        help="Output resolution")
    parser.add_argument("--res_in", type=str, default=None, help="Input resolution key (None for no downscaling)")
    args = parser.parse_args()
    main(args)
