from pathlib import Path
import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from datasets.sr_dataset import SRDataset
from models.attention_edsr import AttentionEDSR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, validation_seed):
    model.eval()
    random_state = random.getstate()
    random.seed(validation_seed)
    total_loss = 0.0

    try:
        for batch in tqdm(loader, desc="Validation"):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            sr = model(lr)
            total_loss += criterion(sr, hr).item()
    finally:
        random.setstate(random_state)

    return total_loss / len(loader)


def load_pretrained_edsr(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()

    matched_state = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    skipped_source = len(source_state) - len(matched_state)
    new_target = len(target_state) - len(matched_state)
    model.load_state_dict(matched_state, strict=False)

    print(f"Pretrained EDSR-lite loaded from: {checkpoint_path}")
    print(f"Matched parameter tensors: {len(matched_state)}")
    print(f"Skipped source tensors: {skipped_source}")
    print(f"Randomly initialized target tensors: {new_target}")


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    train_loss,
    val_loss,
    best_val_loss,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "scale": args.scale,
            "num_features": args.num_features,
            "num_res_blocks": args.num_res_blocks,
            "residual_scale": args.residual_scale,
            "reduction": args.reduction,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Attention-EDSR for x4 SR"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-patch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-edsr",
        type=Path,
        default=None,
        help="EDSR-lite checkpoint used only when no resume checkpoint exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print("=" * 60)
    print("Training Attention-EDSR")
    print(f"Device: {device}")
    print(f"Scale: x{args.scale}")
    print(f"Features / residual blocks: {args.num_features} / {args.num_res_blocks}")
    print(f"Attention reduction: {args.reduction}")
    print(f"Epochs / batch size: {args.epochs} / {args.batch_size}")
    print("=" * 60)

    train_dataset = SRDataset(
        project_root=project_root,
        csv_path=f"data/meta/train_x{args.scale}.csv",
        scale=args.scale,
        training=True,
        lr_patch_size=args.lr_patch_size,
        augment=True,
    )
    # Fixed random patches keep validation deterministic and batchable.
    val_dataset = SRDataset(
        project_root=project_root,
        csv_path=f"data/meta/val_x{args.scale}.csv",
        scale=args.scale,
        training=True,
        lr_patch_size=args.lr_patch_size,
        augment=False,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = AttentionEDSR(
        scale=args.scale,
        num_features=args.num_features,
        num_res_blocks=args.num_res_blocks,
        residual_scale=args.residual_scale,
        reduction=args.reduction,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = project_root / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / f"attention_edsr_x{args.scale}_last.pth"
    best_path = checkpoint_dir / f"attention_edsr_x{args.scale}_best.pth"
    start_epoch = 1
    best_val_loss = float("inf")

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
        print(f"Resuming from epoch {start_epoch}: {last_path}")
        if args.pretrained_edsr is not None:
            print("Resume checkpoint found; --pretrained-edsr was not applied.")
    elif args.pretrained_edsr is not None:
        pretrained_path = args.pretrained_edsr
        if not pretrained_path.is_absolute():
            pretrained_path = project_root / pretrained_path
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained EDSR-lite checkpoint not found: {pretrained_path}"
            )
        load_pretrained_edsr(model, pretrained_path, device)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            validation_seed=args.seed + 1,
        )
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            train_loss,
            val_loss,
            best_val_loss,
            args,
        )
        if is_best:
            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                train_loss,
                val_loss,
                best_val_loss,
                args,
            )
            print(f"Best model saved: {best_path}")

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("=" * 60)
    print("Attention-EDSR training completed.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Total Time: {elapsed_minutes:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
