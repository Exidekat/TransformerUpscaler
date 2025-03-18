"""
train.py

This script instantiates the TransformerModel and trains it on data loaded
using the highres_img_dataset. The dataloader now generates three LR–HR pairs
for each image (720→1080, 720→1440, and 1080→1440). Because the LR–HR pairs
have different resolutions, a custom collate function is used that returns lists
of tensors. The training loop then iterates over each sample in the batch, computes
the loss individually (by passing the desired HR resolution to the model), and averages
the loss before backpropagation.

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

def custom_collate_fn(batch):
    """
    Custom collate function that returns a tuple of lists.
    Each element in the batch is a tuple (lr_tensor, hr_tensor).
    """
    lr_list, hr_list = zip(*batch)
    return list(lr_list), list(hr_list)

def main(args):
    # Device selection: use mps if available, else cuda, otherwise cpu.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Create dataset and DataLoader with custom collate function.
    dataset = highres_img_dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Instantiate the TransformerModel and move it to the device.
    model = TransformerModel().to(device)

    # Define loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (lr_list, hr_list) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_losses = []
            # Process each sample in the batch individually.
            for lr_img, hr_img in zip(lr_list, hr_list):
                # Add batch dimension (1, C, H, W)
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)
                # Determine target resolution from the HR image shape.
                target_res = (hr_img.shape[2], hr_img.shape[3])
                output = model(lr_img, res_out=target_res)
                loss = criterion(output, hr_img)
                batch_losses.append(loss)
            # Average the loss over the batch.
            batch_loss = sum(batch_losses) / len(batch_losses)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(dataloader)}] Loss: {batch_loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint periodically.
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
