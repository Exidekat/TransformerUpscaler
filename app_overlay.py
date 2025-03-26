#!/usr/bin/env python
"""
app_overlay.py

This script provides an interactive window selection and overlay for the Transformer upscaler.
It works cross‑platform:
  - On macOS, it uses Quartz (via PyObjC) to list windows and capture the content of the selected window.
  - On Windows, it uses pygetwindow to list windows and PIL.ImageGrab to capture the window content.
  - On Linux, it falls back to capturing a screen region using mss.

Once the user selects a window, the script continuously captures the target window’s content
(using OS‑specific methods), runs our TransformerModel on the captured image (after resizing to 720×1280),
and displays an overlay (via OpenCV) that is resized to cover the target window.
Additionally, on macOS the overlay window is adjusted upward slightly, and is set to be click‑through.
Press "q" to exit.

Refinements implemented in this version:
  - Asynchronous GPU Streams (CUDA stream) to overlap GPU work.
  - Preallocation of a final output buffer to avoid repeated allocations.
  - Merged post‑processing operations (multiplication, clamping, conversion, permutation, and channel swap)
    using fused PyTorch operations (and an optional Numba kernel for channel swapping).
  - Optional downscaling of captured frames to reduce pipeline load.
  - Reduced frequency of non‑essential operations (cv2.moveWindow is updated every 50 iterations).
  - Parallelization of pre‑processing via a ThreadPoolExecutor.
"""

import importlib
import cv2
import numpy as np
import torch
import time
import argparse
import platform
import mss
import threading
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

# Optional: Numba JIT for custom channel swap kernel.
try:
    from numba import njit, prange
    USE_NUMBA_CHANNEL_SWAP = True
except ImportError:
    USE_NUMBA_CHANNEL_SWAP = False

if platform.system() == "Darwin":
    import Quartz
    from AppKit import NSApplication
elif platform.system() == "Windows":
    import pygetwindow as gw
else:
    import pygetwindow as gw  # Linux fallback

from tools.utils import get_latest_checkpoint, resolutions

# ------------------ Numba custom kernel for channel swapping ------------------
if USE_NUMBA_CHANNEL_SWAP:
    @njit(parallel=True, fastmath=True)
    def numba_channel_swap(arr):
        # arr shape: (H, W, 3) assumed in RGB; swap to BGR
        H, W, C = arr.shape
        out = np.empty_like(arr)
        for i in prange(H):
            for j in range(W):
                out[i, j, 0] = arr[i, j, 2]
                out[i, j, 1] = arr[i, j, 1]
                out[i, j, 2] = arr[i, j, 0]
        return out
# -----------------------------------------------------------------------------

# ------------------ Asynchronous Frame Capture ------------------
class FrameGrabber:
    """
    Continuously captures frames asynchronously using a background thread.
    """
    def __init__(self, capture_func):
        self.capture_func = capture_func
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            frame = self.capture_func()
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
# ------------------------------------------------------------------

# ------------------ Capture Functions ------------------
def list_windows_macos():
    """Retrieve on‑screen windows (as dictionaries) with non‑empty titles using Quartz."""
    window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    windows = []
    for window in window_info_list:
        title = window.get('kCGWindowName', '')
        if title and title.strip() != "":
            windows.append(window)
    return windows

def select_window_macos():
    """List available macOS windows and let the user select one."""
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
    """Extract bounding box as (left, top, width, height) from a Quartz window dict."""
    bounds = window.get('kCGWindowBounds', {})
    left = int(bounds.get('X', 0))
    top = int(bounds.get('Y', 0))
    width = int(bounds.get('Width', 0))
    height = int(bounds.get('Height', 0))
    return left, top, width, height

def capture_window_content_macos(window):
    """Capture macOS window content using Quartz; returns a PIL Image in RGB."""
    from Quartz import CGWindowListCreateImage, kCGWindowListOptionIncludingWindow, kCGWindowImageDefault, CGRectMake
    window_id = window.get('kCGWindowNumber')
    bounds = window.get('kCGWindowBounds', {})
    x = float(bounds.get('X', 0))
    y = float(bounds.get('Y', 0))
    width = float(bounds.get('Width', 0))
    height = float(bounds.get('Height', 0))
    rect = CGRectMake(x, y, width, height)
    cg_image = CGWindowListCreateImage(rect, kCGWindowListOptionIncludingWindow, window_id, kCGWindowImageDefault)
    if cg_image is None:
        return None
    import Quartz
    w = Quartz.CGImageGetWidth(cg_image)
    h = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    data_provider = Quartz.CGImageGetDataProvider(cg_image)
    data = Quartz.CGDataProviderCopyData(data_provider)
    img = Image.frombuffer("RGBA", (w, h), data, "raw", "RGBA", bytes_per_row, 1)
    return img.convert("RGB")

def set_overlay_passthrough_macos(window_title):
    """Set the overlay window to ignore mouse events on macOS."""
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()
    time.sleep(0.5)
    for win in app.windows():
        if window_title in str(win.title()):
            win.setIgnoresMouseEvents_(True)
            print(f"Set window '{win.title()}' to be click-through.")
            return
    print("Could not set overlay window to click-through.")

def select_window_windows():
    """List available windows using pygetwindow and let the user select one."""
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
    class DummyWindow: pass
    win = DummyWindow()
    win.left = left
    win.top = top
    win.width = width
    win.height = height
    return win

def capture_window_content_windows(win):
    """Capture Windows window content using PIL.ImageGrab; returns a PIL Image in RGB."""
    from PIL import ImageGrab
    bbox = (win.left, win.top, win.left + win.width, win.top + win.height)
    return ImageGrab.grab(bbox).convert("RGB")

def capture_window_content_linux(monitor):
    """Fallback: Capture a screen region using mss on Linux."""
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
# -----------------------------------------------------------------

# ------------------ Main App ------------------
def main(args):
    # Optional: define a downscaling factor for captured frames to reduce pipeline load.
    CAPTURE_DOWNSCALE = 0.5  # set to e.g. 0.75 if you want to downscale captured frames

    sys_platform = platform.system()
    if sys_platform == "Darwin":
        selected_window = select_window_macos()
        selected_title = selected_window.get('kCGWindowName', 'No Title')
        print(f"Selected window: {selected_title}")
        left, top, width, height = get_window_bounds_macos(selected_window)
        top = max(0, top - 65)
        capture_func = lambda: capture_window_content_macos(selected_window)
    elif sys_platform == "Windows":
        win = select_window_windows()
        print(f"Selected window bounds: left={win.left}, top={win.top}, width={win.width}, height={win.height}")
        capture_func = lambda: capture_window_content_windows(win)
        left, top, width, height = win.left, win.top, win.width, win.height
    else:
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
        res_in = resolutions[args.res_in]
    else:
        res_in = None

    res_out = resolutions[args.res_out]

    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"models/{args.model}/checkpoints"

    model = TransformerModel().to(device)
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile!")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    if args.quantize:
        print("Applying dynamic quantization to the model...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        print("Model quantization complete.")

    use_amp = device.type in ['cuda', 'mps']

    # Preallocate an output buffer for the final overlay image.
    final_overlay_buffer = np.empty((height, width, 3), dtype=np.uint8)

    # Preallocate a CUDA stream if using CUDA.
    gpu_stream = torch.cuda.Stream() if (device.type == 'cuda') else None

    # Create a transformation for the low-resolution image.
    lr_transform = transforms.Compose([
        transforms.Resize(res_in),
        transforms.ToTensor()
    ]) if res_in is not None else transforms.ToTensor()

    # Use a ThreadPoolExecutor to parallelize pre-processing.
    executor = ThreadPoolExecutor(max_workers=1)
    preproc_future = None

    # Optionally, define a helper function for pre-processing (can be JIT compiled with Numba if desired).
    def preproc_worker(img):
        # If downscaling is desired, do it here.
        if CAPTURE_DOWNSCALE < 1.0:
            new_size = (int(img.width * CAPTURE_DOWNSCALE), int(img.height * CAPTURE_DOWNSCALE))
            img = img.resize(new_size, resample=Image.BILINEAR)
        return lr_transform(img)

    # Start asynchronous frame capture.
    grabber = FrameGrabber(capture_func).start()

    window_name = "Overlay Upscaled"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    if sys_platform == "Darwin":
        time.sleep(0.5)
        set_overlay_passthrough_macos(window_name)

    # Initialize profiling accumulators.
    timings = {
        "capture": 0.0,
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "resize": 0.0,
        "move_window": 0.0,
        "display": 0.0
    }
    iterations = 0
    move_window_interval = 50  # update window position every 50 iterations

    try:
        while True:
            iter_start = time.time()
            iterations += 1

            # Capture step.
            t0 = time.time()
            captured_img = grabber.read()
            t1 = time.time()
            timings["capture"] += (t1 - t0)
            if captured_img is None:
                continue

            # Preprocessing step (parallelized).
            t0 = time.time()
            if preproc_future is None:
                preproc_future = executor.submit(preproc_worker, captured_img)
                continue  # wait for next iteration to get a result
            else:
                lr_tensor = preproc_future.result()
                preproc_future = executor.submit(preproc_worker, captured_img)
            # Add batch dimension and transfer to device.
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            t1 = time.time()
            timings["preprocess"] += (t1 - t0)

            # Inference step with optional asynchronous GPU stream.
            t0 = time.time()
            with torch.no_grad():
                if use_amp:
                    if gpu_stream is not None:
                        with torch.cuda.stream(gpu_stream), torch.autocast(device_type=device.type, dtype=torch.float16):
                            upscaled = model(lr_tensor, res_out=res_out)
                        torch.cuda.current_stream().wait_stream(gpu_stream)
                    else:
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            upscaled = model(lr_tensor, res_out=res_out)
                else:
                    upscaled = model(lr_tensor, res_out=res_out)
            t1 = time.time()
            timings["inference"] += (t1 - t0)

            # Merged postprocessing: perform multiplication, clamping, type conversion,
            # permutation, and channel swap all within torch.
            t0 = time.time()
            upscaled = upscaled.squeeze(0)
            upscaled = (upscaled * 255).clamp(0, 255).to(torch.uint8)
            upscaled = upscaled.permute(1, 2, 0)   # (H, W, C) in RGB
            # Perform channel swap inside torch instead of numpy.
            upscaled = upscaled[..., [2, 1, 0]]  # now BGR
            # Transfer to CPU and convert to numpy array.
            upscaled_np = upscaled.cpu().numpy()
            # Optionally, use Numba kernel to swap channels if USE_NUMBA_CHANNEL_SWAP is enabled.
            if USE_NUMBA_CHANNEL_SWAP:
                upscaled_np = numba_channel_swap(upscaled_np)
            t1 = time.time()
            timings["postprocess"] += (t1 - t0)

            # Resize step using preallocated buffer.
            t0 = time.time()
            resized = cv2.resize(upscaled_np, (width, height))
            np.copyto(final_overlay_buffer, resized)
            overlay = final_overlay_buffer
            t1 = time.time()
            timings["resize"] += (t1 - t0)

            # Reduce frequency of non-essential window repositioning.
            t0 = time.time()
            if iterations % move_window_interval == 0:
                cv2.moveWindow(window_name, left, top)
            t1 = time.time()
            timings["move_window"] += (t1 - t0)

            # Display step.
            t0 = time.time()
            frame_end = time.time()
            fps = 1.0 / (frame_end - iter_start)
            cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow(window_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t1 = time.time()
            timings["display"] += (t1 - t0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Profiling results:")
        for step, total_time in timings.items():
            avg_time = total_time / iterations if iterations > 0 else 0.0
            print(f"{step}: total = {total_time:.4f} sec, average per iteration = {avg_time:.4f} sec")
        max_step = max(timings, key=lambda k: timings[k] / iterations if iterations > 0 else 0)
        max_avg = timings[max_step] / iterations if iterations > 0 else 0.0
        print(f"Step that took the most time on average: {max_step} ({max_avg:.4f} sec per iteration)")

    grabber.stop()
    executor.shutdown(wait=True)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay App for Transformer Upscaler with advanced optimizations."
    )
    parser.add_argument("--model", type=str, default="EfficientTransformer",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--res_out", type=str, default='4k',
                        help="Output resolution key")
    parser.add_argument("--res_in", type=str, default=None,
                        help="Input resolution key (None for no downscaling)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable model compilation with torch.compile")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable dynamic quantization on the model to reduce footprint")
    args = parser.parse_args()
    main(args)
