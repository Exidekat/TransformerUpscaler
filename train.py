#!/usr/bin/env python
"""
train.py

This script instantiates the TransformerModel from the specified model package and trains it on data loaded
using the highres_img_dataset_online. This enables AB testing across models by specifying --model (which determines
the module path models/{args.model}/model.py) and automatically sets the checkpoint directory to
models/{args.model}/checkpoints/ if not provided.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'

import asyncio
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torchvision.transforms import transforms
import importlib
import warnings
from contextlib import nullcontext  # Used as a no-op context manager
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import math

# Import the dataset.
from data_handling.data_class import highres_img_dataset_online, highres_img_dataset
from tools.utils import get_latest_checkpoint

warnings.filterwarnings("ignore", category=FutureWarning)


def custom_collate_fn(batch):
    """
    Custom collate function that returns a tuple of lists.
    Each element in the batch is a tuple (lr_tensor, hr_tensor).
    """
    lr_list, hr_list = zip(*batch)
    return list(lr_list), list(hr_list)

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
    """
    Uses SequentialLR to first apply **linear warm-up**, then **cosine annealing decay**.
    """
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=min_lr)
    
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


def main(args):
    # If no checkpoint directory is provided, use the default based on the model name.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("models", args.model, "checkpoints")

    epochs = args.epochs

    # Dynamically import the model module.
    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    # Device selection: prefer mps if available, then cuda, otherwise cpu.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Setup an autocast context manager.
    # For CUDA, use torch.cuda.amp.autocast.
    # For MPS, try to use torch.amp.autocast with device_type="mps" (if available),
    # otherwise, use nullcontext to effectively disable autocasting.
    if device.type == "cuda":
        amp_autocast = torch.cuda.amp.autocast
    elif device.type == "mps":
        try:
            amp_autocast = lambda: torch.amp.autocast(device_type="mps")
        except Exception:
            amp_autocast = nullcontext
    else:
        amp_autocast = nullcontext

    # Create dataset and DataLoader with custom collate function.
    if args.data_dir is None:
        dataset = highres_img_dataset_online()
    else:
        dataset = highres_img_dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Instantiate the model and move it to the device.
    model = TransformerModel().to(device)

    # Define loss function and optimizer.
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() 
    
    epochs_trained = 0
    warmup_epochs = 3
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=args.epochs)

    try:
        checkpoint_path, loaded_epoch = get_latest_checkpoint(args.checkpoint_dir)
    except FileNotFoundError as e:
        print(e)
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f'Loading checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])  # Restore GradScaler state
            epochs_trained = checkpoint['epoch']
            print(f'Successfully resumed from epoch {epochs_trained}')
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            epochs_trained = 0
    else:
        print(f'No checkpoint found. Starting training from scratch.')
        epochs_trained = 0

    # Import resizing transforms, for downsizing upscaled outputs
    resize_transforms = {}

    torch.cuda.empty_cache()

    model.train()
    for epoch in range(epochs_trained, epochs):
        running_loss = 0.0
        for batch_idx, (lr_list, hr_list) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_losses = []

            # Use our conditional autocast context.
            with amp_autocast():
                # Process each sample in the batch individually.
                for lr_img, hr_img in zip(lr_list, hr_list):
                    # Add batch dimension (1, C, H, W)
                    lr_img = lr_img.unsqueeze(0).to(device)
                    hr_img = hr_img.unsqueeze(0).to(device)

                    output = model(lr_img, res_out=(hr_img.shape[2], hr_img.shape[3]), require_ratio=False)

                    # in case the output isn't the right ratio yet, squash
                    if (output.shape[2], output.shape[3]) != (hr_img.shape[2], hr_img.shape[3]):
                        if (hr_img.shape[2], hr_img.shape[3]) not in resize_transforms.keys():
                            resize_transforms[(hr_img.shape[2], hr_img.shape[3])] = transforms.Resize((hr_img.shape[2], hr_img.shape[3]))
                        output = resize_transforms[(hr_img.shape[2], hr_img.shape[3])](output)

                    loss = criterion(output, hr_img)
                    batch_losses.append(loss)

            # Average the loss over the batch.
            batch_loss = sum(batch_losses) / len(batch_losses)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += batch_loss.item()
            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] Step [{batch_idx + 1}/{len(dataloader)}] Loss: {batch_loss.item():.6f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6e}"
                )
                
        scheduler.step()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] completed. Average Loss: {avg_loss:.6f}")

        # Save model checkpoint periodically.
        if (epoch + 1) % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")

            checkpoint = {
                "epoch": epoch + 1,  # Save next epoch number
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()  # Save GradScaler for AMP
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TransformerModel for image upscaling")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the directory containing training images (.jpg)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Interval (in batches) to log training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save model checkpoint every n epochs")
    parser.add_argument("--model", type=str, default="FastTransformer",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--traceback", action="store_true",
                        help="Enable the Traceback Window") 

    args = parser.parse_args()

    if args.traceback:
        from tools.TracebackWindow import traceback_display


        @traceback_display
        def run():
            asyncio.run(main(args))
    else:
        def run():
            asyncio.run(main(args))
    run()
