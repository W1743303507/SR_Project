from pathlib import Path
import csv
import sys
import time

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models.srcnn import SRCNN
from utils.metrics import calculate_psnr, calculate_ssim


def load_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(tensor):
    arr = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_comparison(lr_img, bicubic_img, sr_img, hr_img):
    lr_up = lr_img.resize(hr_img.size, Image.Resampling.NEAREST)
    w, h = hr_img.size
    canvas = Image.new("RGB", (w * 4, h))
    canvas.paste(lr_up, (0, 0))
    canvas.paste(bicubic_img, (w, 0))
    canvas.paste(sr_img, (w * 2, 0))
    canvas.paste(hr_img, (w * 3, 0))
    return canvas


def measure_inference(model, lr_tensor, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return sr_tensor, time.perf_counter() - start_time


def evaluate_dataset(model, device, csv_file, dataset_name):
    rows = load_csv(project_root / csv_file)
    if not rows:
        raise RuntimeError(f"No samples found in CSV: {csv_file}")

    output_dir = project_root / "results" / "srcnn" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    psnr_values = []
    ssim_values = []
    inference_times = []

    for index, row in enumerate(tqdm(rows, desc=f"SRCNN {dataset_name}")):
        lr_img = Image.open(project_root / row["lr_path"]).convert("RGB")
        hr_img = Image.open(project_root / row["hr_path"]).convert("RGB")
        lr_tensor = pil_to_tensor(lr_img).to(device)

        sr_tensor, inference_time = measure_inference(
            model, lr_tensor, device
        )
        sr_img = tensor_to_pil(sr_tensor)

        if sr_img.size != hr_img.size:
            raise ValueError(
                f"SR-HR size mismatch: SR={sr_img.size}, HR={hr_img.size}, "
                f"image={row['hr_path']}"
            )

        bicubic_img = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
        sr_np = np.asarray(sr_img, dtype=np.uint8)
        hr_np = np.asarray(hr_img, dtype=np.uint8)
        psnr_value = calculate_psnr(sr_np, hr_np)
        ssim_value = calculate_ssim(sr_np, hr_np)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        inference_times.append(inference_time)

        image_name = Path(row["hr_path"]).stem
        sr_img.save(output_dir / f"{image_name}_srcnn.png")

        if index < 5:
            comparison = make_comparison(lr_img, bicubic_img, sr_img, hr_img)
            comparison.save(output_dir / f"{image_name}_comparison.png")

        metric_rows.append(
            {
                "dataset": dataset_name,
                "image": image_name,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "time": inference_time,
            }
        )

    summary = {
        "dataset": dataset_name,
        "avg_psnr": float(np.mean(psnr_values)),
        "avg_ssim": float(np.mean(ssim_values)),
        "avg_time": float(np.mean(inference_times)),
    }

    print("-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Average PSNR: {summary['avg_psnr']:.4f}")
    print(f"Average SSIM: {summary['avg_ssim']:.4f}")
    print(f"Average Time: {summary['avg_time']:.6f} s/image")
    print("-" * 60)

    return metric_rows, summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(device, scale=4):
    checkpoint_path = (
        project_root / "results" / "checkpoints" / "srcnn_x4_best.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = SRCNN(scale=scale).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint loaded: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


def main():
    scale = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_sets = [
        ("Set5", "data/meta/test_set5_x4.csv"),
        ("Set14", "data/meta/test_set14_x4.csv"),
        ("BSD100", "data/meta/test_bsd100_x4.csv"),
        ("Urban100", "data/meta/test_urban100_x4.csv"),
    ]

    print("=" * 60)
    print("Testing SRCNN")
    print(f"Device: {device}")
    print(f"Scale: x{scale}")
    print("=" * 60)

    model = load_model(device=device, scale=scale)
    all_metric_rows = []
    summary_rows = []

    for dataset_name, csv_file in test_sets:
        metric_rows, summary = evaluate_dataset(
            model=model,
            device=device,
            csv_file=csv_file,
            dataset_name=dataset_name,
        )
        all_metric_rows.extend(metric_rows)
        summary_rows.append(summary)

    metrics_dir = project_root / "results" / "metrics"
    details_path = metrics_dir / "srcnn_x4_details.csv"
    summary_path = metrics_dir / "srcnn_x4_summary.csv"

    write_csv(
        details_path,
        all_metric_rows,
        fieldnames=["dataset", "image", "psnr", "ssim", "time"],
    )
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=["dataset", "avg_psnr", "avg_ssim", "avg_time"],
    )

    print("=" * 60)
    print("SRCNN testing completed.")
    print(f"Summary saved to: {summary_path}")
    print(f"Details saved to: {details_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
