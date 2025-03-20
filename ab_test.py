#!/usr/bin/env python
"""
ab_test.py

This script performs an AB test between two models by running inference on every sample
from the dataset (loaded from --data_dir) using both models. It computes the MSE loss
between each model’s output and the ground truth, sums the losses, and prints the total
and average losses for each model.

New optional arguments:
  --res_in: if provided (e.g., 720), only process samples where the LR image has height 720.
  --res_out: if provided (e.g., 1080), only process samples where the HR image has height 1080.

Usage:
    python ab_test.py --data_dir <data_dir> --model_a <modelA_name> --model_b <modelB_name> [--batch_size 1]
                       [--res_in 720] [--res_out 1080]
Default checkpoint directories are assumed to be:
    models/{model_a}/checkpoints and models/{model_b}/checkpoints.
"""

import os
import argparse
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_handling.data_class import highres_img_dataset
from tools.utils import get_latest_checkpoint

# Custom collate function returns a tuple of lists
def custom_collate_fn(batch):
    lr_list, hr_list = zip(*batch)
    return list(lr_list), list(hr_list)

def main(args):
    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running AB test on device: {device}")

    # Set up dataset and DataLoader with custom collate.
    dataset = highres_img_dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=custom_collate_fn)

    # Dynamically import models.
    model_module_a = importlib.import_module(f"models.{args.model_a}.model")
    TransformerModelA = model_module_a.TransformerModel
    model_module_b = importlib.import_module(f"models.{args.model_b}.model")
    TransformerModelB = model_module_b.TransformerModel

    # Set default checkpoint directories if not provided.
    checkpoint_dir_a = args.checkpoint_dir_a or os.path.join("models", args.model_a, "checkpoints")
    checkpoint_dir_b = args.checkpoint_dir_b or os.path.join("models", args.model_b, "checkpoints")

    # Load latest checkpoints.
    ckpt_a, _ = get_latest_checkpoint(checkpoint_dir_a)
    ckpt_b, _ = get_latest_checkpoint(checkpoint_dir_b)
    print(f"Model A ({args.model_a}) checkpoint: {ckpt_a}")
    print(f"Model B ({args.model_b}) checkpoint: {ckpt_b}")

    # Instantiate models and load weights.
    model_a = TransformerModelA().to(device)
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    model_a.eval()

    model_b = TransformerModelB().to(device)
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    model_b.eval()

    # Define loss criterion.
    criterion = nn.MSELoss(reduction="mean")

    total_loss_a = 0.0
    total_loss_b = 0.0
    processed_samples = 0

    # Loop over dataset without gradients.
    with torch.no_grad():
        for batch_idx, (lr_list, hr_list) in enumerate(dataloader):
            for lr_img, hr_img in zip(lr_list, hr_list):
                # Check resolution restrictions if provided.
                if args.res_in is not None and lr_img.shape[1] != args.res_in:
                    continue
                if args.res_out is not None and hr_img.shape[1] != args.res_out:
                    continue

                # Move images to device and add batch dimension.
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)
                # Determine target resolution from HR image.
                target_res = (hr_img.shape[2], hr_img.shape[3])
                # Run inference for both models.
                output_a = model_a(lr_img, res_out=target_res)
                output_b = model_b(lr_img, res_out=target_res)
                # Compute loss.
                loss_a = criterion(output_a, hr_img)
                loss_b = criterion(output_b, hr_img)
                total_loss_a += loss_a.item()
                total_loss_b += loss_b.item()
                processed_samples += 1
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Processed {processed_samples} samples so far...")

    if processed_samples == 0:
        print("No samples matched the specified resolution criteria.")
        return

    avg_loss_a = total_loss_a / processed_samples
    avg_loss_b = total_loss_b / processed_samples

    print("========================================")
    print(f"Model A ({args.model_a}) Total Loss: {total_loss_a:.4f} | Average Loss: {avg_loss_a:.4f}")
    print(f"Model B ({args.model_b}) Total Loss: {total_loss_b:.4f} | Average Loss: {avg_loss_b:.4f}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AB Test for Transformer Upscaler Models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing images (.jpg)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of samples per iteration)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log progress every N batches")
    parser.add_argument("--model_a", type=str, required=True,
                        help="Model A name (e.g., 'ResidualTransformer' or 'HierarchicalTransformer')")
    parser.add_argument("--model_b", type=str, required=True,
                        help="Model B name")
    parser.add_argument("--checkpoint_dir_a", type=str, default=None,
                        help="Checkpoint directory for model A (default: models/{model_a}/checkpoints/)")
    parser.add_argument("--checkpoint_dir_b", type=str, default=None,
                        help="Checkpoint directory for model B (default: models/{model_b}/checkpoints/)")
    parser.add_argument("--res_in", type=int, default=None,
                        help="Restrict testing to only LR images with this vertical resolution (e.g., 720)")
    parser.add_argument("--res_out", type=int, default=None,
                        help="Restrict testing to only HR images with this vertical resolution (e.g., 1080)")
    args = parser.parse_args()
    main(args)
