from pathlib import Path
import sys
import csv
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.metrics import calculate_psnr, calculate_ssim


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def pil_to_numpy(img):
    return np.array(img).astype(np.uint8)


def make_comparison(lr_img, sr_img, hr_img):
    """
    Create side-by-side image:
    LR upsampled | Bicubic SR | HR
    """
    lr_up = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)

    w, h = hr_img.size
    canvas = Image.new("RGB", (w * 3, h))

    canvas.paste(lr_up, (0, 0))
    canvas.paste(sr_img, (w, 0))
    canvas.paste(hr_img, (w * 2, 0))

    return canvas


def evaluate_dataset(project_root, csv_file, dataset_name, scale=4):
    rows = load_csv(project_root / csv_file)

    output_dir = project_root / "results" / "bicubic" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []

    psnr_list = []
    ssim_list = []
    time_list = []

    for idx, row in enumerate(tqdm(rows, desc=f"Bicubic {dataset_name}")):
        lr_path = project_root / row["lr_path"]
        hr_path = project_root / row["hr_path"]

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        start_time = time.time()
        sr_img = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
        infer_time = time.time() - start_time

        sr_np = pil_to_numpy(sr_img)
        hr_np = pil_to_numpy(hr_img)

        psnr_value = calculate_psnr(sr_np, hr_np)
        ssim_value = calculate_ssim(sr_np, hr_np)

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        time_list.append(infer_time)

        image_name = Path(row["hr_path"]).stem

        sr_save_path = output_dir / f"{image_name}_bicubic.png"
        sr_img.save(sr_save_path)

        # Save comparison for first 5 images
        if idx < 5:
            comparison = make_comparison(lr_img, sr_img, hr_img)
            comparison_save_path = output_dir / f"{image_name}_comparison.png"
            comparison.save(comparison_save_path)

        metric_rows.append({
            "dataset": dataset_name,
            "image": image_name,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "time": infer_time,
        })

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_time = float(np.mean(time_list))

    print("-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Time: {avg_time:.6f} s/image")
    print("-" * 60)

    return metric_rows, {
        "dataset": dataset_name,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_time": avg_time,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    test_sets = [
        {
            "name": "Set5",
            "csv": "data/meta/test_set5_x4.csv",
        },
        {
            "name": "Set14",
            "csv": "data/meta/test_set14_x4.csv",
        },
        {
            "name": "BSD100",
            "csv": "data/meta/test_bsd100_x4.csv",
        },
        {
            "name": "Urban100",
            "csv": "data/meta/test_urban100_x4.csv",
        },
    ]

    all_metric_rows = []
    summary_rows = []

    for item in test_sets:
        metric_rows, summary = evaluate_dataset(
            project_root=project_root,
            csv_file=item["csv"],
            dataset_name=item["name"],
            scale=4,
        )

        all_metric_rows.extend(metric_rows)
        summary_rows.append(summary)

    write_csv(
        project_root / "results" / "metrics" / "bicubic_x4_details.csv",
        all_metric_rows,
        fieldnames=["dataset", "image", "psnr", "ssim", "time"]
    )

    write_csv(
        project_root / "results" / "metrics" / "bicubic_x4_summary.csv",
        summary_rows,
        fieldnames=["dataset", "avg_psnr", "avg_ssim", "avg_time"]
    )

    print("=" * 60)
    print("Bicubic baseline completed.")
    print("Results saved to:")
    print(project_root / "results" / "metrics" / "bicubic_x4_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()