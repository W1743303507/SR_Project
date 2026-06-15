from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from datasets.sr_dataset import SRDataset
from models.srcnn import SRCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation"):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        sr = model(lr)
        loss = criterion(sr, hr)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    scale = 4
    epochs = 30
    batch_size = 8
    lr_patch_size = 48
    learning_rate = 1e-4
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Training SRCNN")
    print(f"Device: {device}")
    print(f"Scale: x{scale}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    train_dataset = SRDataset(
        project_root=project_root,
        csv_path="data/meta/train_x4.csv",
        scale=scale,
        training=True,
        lr_patch_size=lr_patch_size,
        augment=True,
    )

    val_dataset = SRDataset(
        project_root=project_root,
        csv_path="data/meta/val_x4.csv",
        scale=scale,
        training=True,
        lr_patch_size=lr_patch_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = SRCNN(scale=scale).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = project_root / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1

    last_path = checkpoint_dir / "srcnn_x4_last.pth"
    best_path = checkpoint_dir / "srcnn_x4_best.pth"

    if last_path.exists():
        print(f"Resume checkpoint found: {last_path}")
        checkpoint = torch.load(last_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location=device)
            best_val_loss = best_checkpoint.get("val_loss", float("inf"))
        else:
            best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Resume training from epoch {start_epoch}")
        print(f"Current best val loss: {best_val_loss:.6f}")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        last_path = checkpoint_dir / "srcnn_x4_last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "scale": scale,
            },
            last_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = checkpoint_dir / "srcnn_x4_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "scale": scale,
                },
                best_path,
            )

            print(f"Best model saved: {best_path}")

    total_time = time.time() - start_time

    print("=" * 60)
    print("SRCNN training completed.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()