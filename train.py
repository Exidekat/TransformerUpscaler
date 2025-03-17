"""
train.py

This script instantiates the TransformerModel and trains it on data loaded
using the highres_img_dataset. The low resolution images (720×1280) are fed
to the network and the output (1080×1920) is compared with the high resolution target.
Hyperparameters can be adjusted via command-line arguments.
"""

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the dataset and Transformer model
from data_handling.data_class import highres_img_dataset
from model.TransformerModel import TransformerModel

def main(args):
    # if no gpu available, use cpu. if on macos>=13.0, use mps
    DEVICE = "cpu"

    if torch.backends.mps.is_built():
        DEVICE = "mps"
    elif torch.backends.cuda.is_built():
        DEVICE = "cuda"

    device = torch.device(DEVICE)
    print(f"Training on device: {device}")

    # Create dataset and DataLoader
    dataset = highres_img_dataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Instantiate the TransformerModel and move it to the device
    model = TransformerModel().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            # Forward pass: generate upscaled output from low resolution images
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint periodically
        if (epoch + 1) % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TransformerModel for image upscaling")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing training images (.jpg)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval (in batches) to log training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save model checkpoint every n epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    main(args)
