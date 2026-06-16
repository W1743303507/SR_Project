import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import importlib
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 4

CHECKPOINTS = {
    "SRCNN": "results/checkpoints/srcnn_x4_best.pth",
    "EDSR-lite": "results/checkpoints/edsr_lite_x4_best.pth",
    "Attention-EDSR": "results/checkpoints/attention_edsr_x4_best.pth",
}


MODEL_MODULES = {
    "SRCNN": "models.srcnn",
    "EDSR-lite": "models.edsr_lite",
    "Attention-EDSR": "models.attention_edsr",
}


def find_model_class(module_name):
    module = importlib.import_module(module_name)

    candidates = []

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"No torch.nn.Module class found in {module_name}")

    return candidates[0]


def build_model(model_name):
    module_name = MODEL_MODULES[model_name]
    model_class = find_model_class(module_name)

    try:
        model = model_class(scale=SCALE)
    except TypeError:
        try:
            model = model_class(upscale_factor=SCALE)
        except TypeError:
            try:
                model = model_class()
            except TypeError as e:
                raise RuntimeError(f"Cannot initialize {model_name}: {e}")

    return model


def load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def bicubic_sr(lr_img):
    w, h = lr_img.size
    return lr_img.resize((w * SCALE, h * SCALE), Image.BICUBIC)


def run_model(model, lr_img):
    x = pil_to_tensor(lr_img)

    with torch.no_grad():
        start = time.time()
        y = model(x)
        elapsed = time.time() - start

    sr_img = tensor_to_pil(y)
    return sr_img, elapsed


def get_csv_columns(df):
    lower_map = {c.lower(): c for c in df.columns}

    lr_candidates = ["lr_path", "lr", "low_resolution", "lr_image"]
    hr_candidates = ["hr_path", "hr", "high_resolution", "hr_image"]
    name_candidates = ["filename", "image_name", "name", "file_name"]

    lr_col = None
    hr_col = None
    name_col = None

    for c in lr_candidates:
        if c in lower_map:
            lr_col = lower_map[c]
            break

    for c in hr_candidates:
        if c in lower_map:
            hr_col = lower_map[c]
            break

    for c in name_candidates:
        if c in lower_map:
            name_col = lower_map[c]
            break

    if lr_col is None or hr_col is None:
        raise ValueError(
            f"Cannot find LR/HR columns in CSV. Current columns: {list(df.columns)}"
        )

    return lr_col, hr_col, name_col


def add_label(img, label):
    label_h = 36
    w, h = img.size

    canvas = Image.new("RGB", (w, h + label_h), "white")
    canvas.paste(img, (0, label_h))

    draw = ImageDraw.Draw(canvas)
    text_x = 10
    text_y = 8
    draw.text((text_x, text_y), label, fill="black")

    return canvas


def make_grid(images, labels):
    labeled = [add_label(img, label) for img, label in zip(images, labels)]

    heights = [img.height for img in labeled]
    max_h = max(heights)

    normalized = []

    for img in labeled:
        if img.height != max_h:
            new_w = int(img.width * max_h / img.height)
            img = img.resize((new_w, max_h), Image.BICUBIC)
        normalized.append(img)

    total_w = sum(img.width for img in normalized)
    grid = Image.new("RGB", (total_w, max_h), "white")

    x = 0
    for img in normalized:
        grid.paste(img, (x, 0))
        x += img.width

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Set5")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/final_comparisons")
    args = parser.parse_args()

    test_csv = f"data/meta/test_{args.dataset.lower()}_x4.csv"

    if args.dataset == "BSD100":
        test_csv = "data/meta/test_bsd100_x4.csv"
    elif args.dataset == "Urban100":
        test_csv = "data/meta/test_urban100_x4.csv"
    elif args.dataset == "Set14":
        test_csv = "data/meta/test_set14_x4.csv"
    elif args.dataset == "Set5":
        test_csv = "data/meta/test_set5_x4.csv"

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Cannot find test CSV: {test_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Loading CSV: {test_csv}")

    df = pd.read_csv(test_csv)
    lr_col, hr_col, name_col = get_csv_columns(df)

    models = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"Loading {model_name} from {ckpt_path}")
        model = build_model(model_name)
        model = load_checkpoint(model, ckpt_path)
        models[model_name] = model

    selected = df.head(args.max_images)

    for idx, row in selected.iterrows():
        lr_path = row[lr_col]
        hr_path = row[hr_col]

        if not os.path.isabs(lr_path):
            lr_path = os.path.join(os.getcwd(), lr_path)

        if not os.path.isabs(hr_path):
            hr_path = os.path.join(os.getcwd(), hr_path)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        bicubic_img = bicubic_sr(lr_img)

        result_images = [
            bicubic_img,
        ]

        labels = [
            "Bicubic",
        ]

        for model_name, model in models.items():
            sr_img, elapsed = run_model(model, lr_img)
            result_images.append(sr_img)
            labels.append(f"{model_name}")

        result_images.append(hr_img)
        labels.append("HR")

        grid = make_grid(result_images, labels)

        if name_col is not None:
            base_name = str(row[name_col])
        else:
            base_name = os.path.splitext(os.path.basename(hr_path))[0]

        save_name = f"{args.dataset}_{base_name}_comparison.png"
        save_path = os.path.join(args.output_dir, save_name)

        grid.save(save_path)
        print(f"Saved: {save_path}")

    print("Visual comparison images completed.")


if __name__ == "__main__":
    main()