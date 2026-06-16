from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"

SUMMARY_FILES = {
    "Bicubic": "bicubic_x4_summary.csv",
    "SRCNN": "srcnn_x4_summary.csv",
    "EDSR-lite": "edsr_lite_x4_summary.csv",
    "Attention-EDSR": "attention_edsr_x4_summary.csv",
}

DATASET_ORDER = ["Set5", "Set14", "BSD100", "Urban100"]
MODEL_ORDER = ["Bicubic", "SRCNN", "EDSR-lite", "Attention-EDSR"]


def read_model_summary(model_name, file_name):
    path = METRIC_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Cannot find: {path}")

    df = pd.read_csv(path)

    required_columns = {"dataset", "avg_psnr", "avg_ssim", "avg_time"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df["model"] = model_name

    return df[["model", "dataset", "avg_psnr", "avg_ssim", "avg_time"]]


def main():
    all_results = []

    for model_name, file_name in SUMMARY_FILES.items():
        df = read_model_summary(model_name, file_name)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    final_df["model"] = pd.Categorical(
        final_df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    final_df["dataset"] = pd.Categorical(
        final_df["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )

    final_df = final_df.sort_values(["model", "dataset"])

    final_summary_path = METRIC_DIR / "final_summary.csv"
    final_df.to_csv(final_summary_path, index=False, encoding="utf-8-sig")

    psnr_table = final_df.pivot(index="model", columns="dataset", values="avg_psnr")
    ssim_table = final_df.pivot(index="model", columns="dataset", values="avg_ssim")

    psnr_table = psnr_table.reindex(index=MODEL_ORDER, columns=DATASET_ORDER)
    ssim_table = ssim_table.reindex(index=MODEL_ORDER, columns=DATASET_ORDER)

    psnr_table["Average"] = psnr_table.mean(axis=1)
    ssim_table["Average"] = ssim_table.mean(axis=1)

    psnr_table_path = METRIC_DIR / "final_psnr_table.csv"
    ssim_table_path = METRIC_DIR / "final_ssim_table.csv"

    psnr_table.to_csv(psnr_table_path, encoding="utf-8-sig")
    ssim_table.to_csv(ssim_table_path, encoding="utf-8-sig")

    edsr = final_df[final_df["model"] == "EDSR-lite"].set_index("dataset")
    attention = final_df[final_df["model"] == "Attention-EDSR"].set_index("dataset")

    improvement_rows = []

    for dataset in DATASET_ORDER:
        psnr_gain = attention.loc[dataset, "avg_psnr"] - edsr.loc[dataset, "avg_psnr"]
        ssim_gain = attention.loc[dataset, "avg_ssim"] - edsr.loc[dataset, "avg_ssim"]

        improvement_rows.append(
            {
                "dataset": dataset,
                "psnr_gain": psnr_gain,
                "ssim_gain": ssim_gain,
            }
        )

    improvement_df = pd.DataFrame(improvement_rows)

    avg_row = pd.DataFrame(
        [
            {
                "dataset": "Average",
                "psnr_gain": improvement_df["psnr_gain"].mean(),
                "ssim_gain": improvement_df["ssim_gain"].mean(),
            }
        ]
    )

    improvement_df = pd.concat([improvement_df, avg_row], ignore_index=True)

    improvement_path = METRIC_DIR / "attention_vs_edsr_improvement.csv"
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Final result files generated:")
    print(final_summary_path)
    print(psnr_table_path)
    print(ssim_table_path)
    print(improvement_path)
    print("=" * 80)

    print("\nFinal Summary:")
    print(final_df)

    print("\nPSNR Table:")
    print(psnr_table.round(4))

    print("\nSSIM Table:")
    print(ssim_table.round(4))

    print("\nAttention-EDSR Improvement:")
    print(improvement_df.round(4))


if __name__ == "__main__":
    main()