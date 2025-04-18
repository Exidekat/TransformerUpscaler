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

import itertools
import lpips

# Import the dataset.
from data_handling.data_class import highres_img_dataset_online, highres_img_dataset
from tools.utils import get_latest_checkpoint

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    # --- Basic Setup ---
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("models", args.model, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    epochs = args.epochs

    import_safe_model_arg = str(args.model).replace("/", '.')
    model_module = importlib.import_module(f"models.{import_safe_model_arg}.model")
    TransformerModel = model_module.TransformerModel

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # --- Autocast Setup ---
    if device.type == "cuda":
        amp_autocast = torch.cuda.amp.autocast
    elif device.type == "mps":
        try:
            amp_autocast = lambda: torch.amp.autocast(device_type="mps")
        except Exception:
            amp_autocast = nullcontext
    else:
        amp_autocast = nullcontext

    # --- Dataset and DataLoader Setup ---
    # Define all the scale pairs you want to train on
    scale_pairs = [
        {"lr": (48, 48), "hr": (96, 96), "factor": 2},
        {"lr": (48, 48), "hr": (144, 144), "factor": 3},
        {"lr": (48, 48), "hr": (192, 192), "factor": 4},
        {"lr": (48, 48), "hr": (288, 288), "factor": 6}
        # {"lr": (720, 1280), "hr": (2160, 3840)} # Example for non-square  
    ]

    if args.data_dir is None: 
        raise ValueError("Must provide --data_dir for highres_img_dataset")
        # dataset = highres_img_dataset_online() # Keep if you have an online version
    else:
        datasets = {
            pair["factor"]: highres_img_dataset(args.data_dir, scale_pair=pair)
            for pair in scale_pairs
        }
        dataloaders = {
            factor: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True, # Shuffle within each scale factor dataset
                num_workers=args.num_workers, # Use arg for num_workers
                pin_memory=True,
                # drop_last=True # Consider if uneven batch sizes cause issues
            )
            for factor, dataset in datasets.items()
        }
        # Make iterators that cycle indefinitely for simpler epoch looping
        dataloader_iters = {
            factor: itertools.cycle(loader)
            for factor, loader in dataloaders.items()
        } 
        # Calculate total steps per epoch roughly - assumes datasets are same size
        # Or just define an epoch by a fixed number of steps
        steps_per_epoch = sum(len(dl) for dl in dataloaders.values())
        print(f"Total estimated steps per epoch: {steps_per_epoch}")


    # --- Model, Optimizer, Scaler, Scheduler ---
    model = TransformerModel().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    
    if args.use_perceptual:
        lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
    else:
        lpips_loss = None
    
    initial_lr = args.lr if (args.lr is not None) else 2e-4
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    scaler = GradScaler(enabled=(device.type == 'cuda')) # Only enable for CUDA
    epochs_trained = 0
    # Adjust total_epochs for scheduler if you define epoch differently
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    # --- Checkpoint Loading ---
    try:
        checkpoint_path, loaded_epoch = get_latest_checkpoint(args.checkpoint_dir)
    except FileNotFoundError as e:
        print(e)
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Support checkpoints saved as {'model_state_dict': ...} or raw state_dict
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])  # Restore GradScaler state
            except Exception as e:
                print(f"Couldn't load optimizer/scheduler/scaler dicts: {e}")
            
            if args.lr is not None:
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                print(f"Overrode checkpoint LR; now using lr={args.lr:.2e}")
            
            epochs_trained = checkpoint['epoch']
            print(f'Successfully resumed from epoch {epochs_trained}')
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            epochs_trained = 0
    else:
        print(f'No checkpoint found. Starting training from scratch.')
        epochs_trained = 0


    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # --- Training Loop ---
    model.train()
    global_step = 0 # Track total steps for potential fixed-step epochs
    factors_to_train = list(dataloaders.keys())

    for epoch in range(epochs_trained, epochs):
        running_loss = 0.0
        running_l1_loss = 0.0
        running_perceptual_loss = 0.0   
        num_batches_processed_this_epoch = 0


        # Interleave batches from different scale factors (using cycled iterators)
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        for step in range(steps_per_epoch):
            # Choose which scale factor to train on this step
            # Simple round-robin:
            current_factor = factors_to_train[step % len(factors_to_train)]
            dataloader_iter = dataloader_iters[current_factor]

            try:
                lr_batch, hr_batch = next(dataloader_iter)
            except StopIteration:
                # Should not happen with itertools.cycle, but good practice
                print(f"Warning: Dataloader for factor {current_factor} exhausted unexpectedly.")
                continue # Or re-initialize: dataloader_iters[current_factor] = itertools.cycle(dataloaders[current_factor])


            optimizer.zero_grad()

            # Move the entire batch to the device
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            # Pass the scale factor to the model
            target_h, target_w = hr_batch.shape[2], hr_batch.shape[3]

            with amp_autocast():
                 # Update model forward pass to accept upscale_factor
                 # The model now needs to know which upsampling path to use
                 output_batch = model(lr_batch, upscale_factor=current_factor, require_ratio=False) # Pass factor

                 # Optional: Check if output size matches target (should if model handles factor correctly)
                 if (output_batch.shape[2], output_batch.shape[3]) != (target_h, target_w):
                      print(f"Warning: Output size {output_batch.shape[2:]} != target size {(target_h, target_w)} for factor {current_factor}. Resizing.")
                      # This indicates an issue in the model's Upsampler or forward logic
                      output_batch = transforms.functional.resize(output_batch, (target_h, target_w), antialias=True) # Use functional resize

                #  loss = criterion(output_batch, hr_batch)
                
                 l1_loss = criterion_l1(output_batch, hr_batch)

            if args.use_perceptual:
                with torch.no_grad():
                    output_lpips = output_batch * 2.0 - 1.0  # Rescale to [-1, 1]
                    target_lpips = hr_batch * 2.0 - 1.0
                    perceptual_loss = lpips_loss(output_lpips, target_lpips).mean()
                loss = l1_loss + args.lpips_weight * perceptual_loss
            else:
                perceptual_loss = torch.tensor(0.0)  # For logging consistency
                loss = l1_loss

            # Scale, Backward, Step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            running_loss += current_loss
            
            l1_loss_value = l1_loss.item()
            running_l1_loss += l1_loss_value
            
            perceptual_loss_value = perceptual_loss.item()
            running_perceptual_loss += perceptual_loss_value
            
            global_step += 1
            num_batches_processed_this_epoch += 1

            if step % args.log_interval == 0:
                 print(
                    f"Epoch [{epoch + 1}/{args.epochs}] Step [{step + 1}/{steps_per_epoch}] Factor [x{current_factor}]\n\
                    \tLoss: {current_loss:.6f}, L1 Loss: {l1_loss_value:.6f}, Perceptual Loss: {perceptual_loss_value:.6f}, LR: {optimizer.param_groups[0]['lr']:.6e}"
                 )

        # -- End of Epoch --
        # Step the scheduler AFTER processing all batches for the epoch
        scheduler.step()

        avg_loss = running_loss / num_batches_processed_this_epoch if num_batches_processed_this_epoch > 0 else 0.0
        avg_l1_loss = running_l1_loss / num_batches_processed_this_epoch if num_batches_processed_this_epoch > 0 else 0.0
        avg_perceptual_loss = running_perceptual_loss / num_batches_processed_this_epoch if num_batches_processed_this_epoch > 0 else 0.0
        print(f"Epoch [{epoch + 1}/{args.epochs}] completed. Average Loss: {avg_loss:.6f}, L1 Loss: {avg_l1_loss:.6f}, Perceptual Loss: {avg_perceptual_loss:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_interval == 0:
            # ... (checkpoint saving code remains the same, ensure scheduler state is saved) ...
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), # SAVE scheduler state
                "scaler_state_dict": scaler.state_dict()
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TransformerModel for image upscaling")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the directory containing training images (.jpg)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Interval (in batches) to log training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save model checkpoint every n epochs")
    parser.add_argument("--model", type=str, default="Fastv2",
                        help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--traceback", action="store_true",
                        help="Enable the Traceback Window")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of DataLoader workers")
    parser.add_argument("--lpips_weight", type=float, default=0.4,
                        help="Weight for LPIPS loss")
    parser.add_argument("--use_perceptual", action="store_true",
                    help="Include LPIPS perceptual loss using VGG backbone")

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
