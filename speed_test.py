"""
speed_test.py

This script runs inference on all images in the specified --data_dir one at a time
(no batching) using the provided dataloader and TransformerModel. It measures the
total inference time (sum of per-image inference times) as well as the overall wall-clock
time, then prints the average inference time per image.
"""
import importlib
import time
import argparse
import torch
import warnings
from data_handling.data_class import highres_img_dataset
from tools.utils import get_latest_checkpoint

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):
    # if no gpu available, use cpu. if on macos>=13.0, use mps
    DEVICE = "cpu"

    if torch.backends.mps.is_built():
        DEVICE = "mps"
    elif torch.backends.cuda.is_built():
        DEVICE = "cuda"

    device = torch.device(DEVICE)
    print(f"Running speed test on device: {device}")

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

    # Create the dataset and dataloader (batch_size=1 for single image processing)
    dataset = highres_img_dataset(args.data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_images = len(dataset)
    total_inference_time = 0.0
    print(f"Processing {total_images} images...")

    # Measure overall wall-clock time (including any overhead)
    overall_start_time = time.time()

    # Inference loop: time each image individually
    with torch.no_grad():
        for idx, (lr_img, _) in enumerate(dataloader):
            lr_img = lr_img.to(device)
            start_time = time.time()
            _ = model(lr_img, res_out=(2160, 3840))
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    average_time = total_inference_time / total_images if total_images > 0 else 0.0

    print(f"Total inference time (sum over images): {total_inference_time:.4f} seconds")
    print(f"Overall wall-clock time: {overall_time:.4f} seconds")
    print(f"Average inference time per image: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speed test for Transformer upscaler inference")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing images for inference")
    parser.add_argument("--model", type=str, default="StrippedTransformer",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    
    args = parser.parse_args()
    main(args)
