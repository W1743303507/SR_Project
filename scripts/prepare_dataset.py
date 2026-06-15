from pathlib import Path
import argparse
import csv
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images(folder: Path):
    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return sorted(images)


def crop_to_scale(img: Image.Image, scale: int):
    w, h = img.size
    new_w = w - (w % scale)
    new_h = h - (h % scale)
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return img


def make_lr(img: Image.Image, scale: int):
    w, h = img.size
    lr_w = w // scale
    lr_h = h // scale
    return img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)


def relative_path(path: Path, root: Path):
    return path.relative_to(root).as_posix()


def process_images(project_root, src_dir, dataset_name, split, scale, output_root, overwrite=False):
    images = find_images(src_dir)

    if len(images) == 0:
        raise RuntimeError(f"No images found in: {src_dir}")

    rows = []

    if split == "test":
        hr_out_dir = output_root / f"x{scale}" / "test" / dataset_name / "HR"
        lr_out_dir = output_root / f"x{scale}" / "test" / dataset_name / "LR"
    else:
        hr_out_dir = output_root / f"x{scale}" / split / "HR"
        lr_out_dir = output_root / f"x{scale}" / split / "LR"

    hr_out_dir.mkdir(parents=True, exist_ok=True)
    lr_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc=f"{dataset_name} -> {split}"):
        out_name = f"{dataset_name}_{img_path.stem}.png"

        hr_out = hr_out_dir / out_name
        lr_out = lr_out_dir / out_name

        if overwrite or not (hr_out.exists() and lr_out.exists()):
            img = Image.open(img_path).convert("RGB")
            img = crop_to_scale(img, scale)
            lr_img = make_lr(img, scale)

            img.save(hr_out)
            lr_img.save(lr_out)

        rows.append({
            "lr_path": relative_path(lr_out, project_root),
            "hr_path": relative_path(hr_out, project_root),
            "dataset": dataset_name,
            "split": split,
            "scale": scale,
        })

    return rows


def write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lr_path", "hr_path", "dataset", "split", "scale"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    output_root = project_root / "data" / "processed"
    meta_root = project_root / "data" / "meta"

    scale = args.scale

    datasets = {
        "train": [
            {
                "name": "DIV2K",
                "path": raw_root / "DIV2K" / "DIV2K_train_HR",
            },
            {
                "name": "Flickr2K",
                "path": raw_root / "Flickr2K" / "Flickr2K_HR",
            },
        ],
        "val": [
            {
                "name": "DIV2K_valid",
                "path": raw_root / "DIV2K" / "DIV2K_valid_HR",
            },
        ],
        "test": [
            {
                "name": "Set5",
                "path": raw_root / "Set5" / "HR",
            },
            {
                "name": "Set14",
                "path": raw_root / "Set14" / "HR",
            },
            {
                "name": "BSD100",
                "path": raw_root / "BSD100" / "HR",
            },
            {
                "name": "Urban100",
                "path": raw_root / "Urban100" / "HR",
            },
        ],
    }

    train_rows = []
    val_rows = []

    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Scale: x{scale}")
    print("=" * 60)

    for item in datasets["train"]:
        src = item["path"]
        if not src.exists():
            raise FileNotFoundError(f"Missing folder: {src}")

        rows = process_images(
            project_root=project_root,
            src_dir=src,
            dataset_name=item["name"],
            split="train",
            scale=scale,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        train_rows.extend(rows)

    for item in datasets["val"]:
        src = item["path"]
        if not src.exists():
            raise FileNotFoundError(f"Missing folder: {src}")

        rows = process_images(
            project_root=project_root,
            src_dir=src,
            dataset_name=item["name"],
            split="val",
            scale=scale,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        val_rows.extend(rows)

    write_csv(meta_root / f"train_x{scale}.csv", train_rows)
    write_csv(meta_root / f"val_x{scale}.csv", val_rows)

    for item in datasets["test"]:
        src = item["path"]
        if not src.exists():
            raise FileNotFoundError(f"Missing folder: {src}")

        rows = process_images(
            project_root=project_root,
            src_dir=src,
            dataset_name=item["name"],
            split="test",
            scale=scale,
            output_root=output_root,
            overwrite=args.overwrite,
        )

        write_csv(meta_root / f"test_{item['name'].lower()}_x{scale}.csv", rows)

    print("=" * 60)
    print("Dataset preparation completed.")
    print(f"Train images: {len(train_rows)}")
    print(f"Val images: {len(val_rows)}")
    print("Test CSV files generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()