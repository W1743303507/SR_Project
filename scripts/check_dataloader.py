from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from datasets.sr_dataset import SRDataset


def tensor_to_image(tensor):
    """
    Convert tensor [C, H, W] to PIL image.
    """
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = tensor.numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def main():
    project_root = Path(__file__).resolve().parents[1]

    train_dataset = SRDataset(
        project_root=project_root,
        csv_path="data/meta/train_x4.csv",
        scale=4,
        training=True,
        lr_patch_size=48,
        augment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(iter(train_loader))

    lr = batch["lr"]
    hr = batch["hr"]

    print("=" * 60)
    print("DataLoader check passed.")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"LR batch shape: {lr.shape}")
    print(f"HR batch shape: {hr.shape}")
    print(f"Example LR path: {batch['lr_path'][0]}")
    print(f"Example HR path: {batch['hr_path'][0]}")
    print("=" * 60)

    # Save one visual sample for checking
    output_dir = project_root / "results" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_img = tensor_to_image(lr[0])
    hr_img = tensor_to_image(hr[0])

    # Enlarge LR to HR size only for visual comparison
    lr_up = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (hr_img.width * 2, hr_img.height))
    canvas.paste(lr_up, (0, 0))
    canvas.paste(hr_img, (hr_img.width, 0))

    save_path = output_dir / "dataloader_sample.png"
    canvas.save(save_path)

    print(f"Debug image saved to: {save_path}")


if __name__ == "__main__":
    main()