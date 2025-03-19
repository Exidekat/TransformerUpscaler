#!/usr/bin/env python
"""
train.py

This script instantiates the TransformerModel from the specified model package and trains it
on data loaded using highres_img_dataset. Mixed precision is enabled on CUDA via torch.autocast and GradScaler.
An optional flag (--compile) compiles the model using torch.compile (if supported).
On non-CUDA devices, training runs in full precision.
"""

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import importlib
from data_handling.data_class import highres_img_dataset
from tools.utils import get_latest_checkpoint

def custom_collate_fn(batch):
    lr_list, hr_list = zip(*batch)
    return list(lr_list), list(hr_list)

def main(args):

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("models", args.model, "checkpoints")
    epochs = args.epochs

    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    # Device selection.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    dataset = highres_img_dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    model = TransformerModel().to(device).float()
    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile!")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    try:
        checkpoint_path, epochs_trained = get_latest_checkpoint(args.checkpoint_dir)
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        if epochs_trained >= epochs:
            print(f"Checkpoint {checkpoint_path} exceeds epochs {epochs}")
            exit(1)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        epochs_trained = 0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type != "mps"))

    model.train()
    for epoch in range(epochs_trained, epochs):
        running_loss = 0.0
        for batch_idx, (lr_list, hr_list) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_losses = []
            for lr_img, hr_img in zip(lr_list, hr_list):
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)
                target_res = (hr_img.shape[2], hr_img.shape[3])
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(lr_img, res_out=target_res)
                    loss = criterion(output, hr_img)
                batch_losses.append(loss)
            batch_loss = sum(batch_losses) / len(batch_losses)
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += batch_loss.item()
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(dataloader)}] Loss: {batch_loss.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")
        if (epoch + 1) % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TransformerModel for image upscaling with Mixed Precision")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing training images (.jpg)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval (in batches) to log training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model checkpoint every n epochs")
    parser.add_argument("--model", type=str, default="ResidualTransformer", help="Model name to use (corresponds to models/{model}/model.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save model checkpoints (default: models/{model}/checkpoints/)")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation with torch.compile")
    args = parser.parse_args()
    main(args)
