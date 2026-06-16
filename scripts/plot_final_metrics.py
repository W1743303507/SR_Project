import os
import pandas as pd
import matplotlib.pyplot as plt


METRIC_DIR = "results/metrics"
FIG_DIR = "results/figures"


def plot_metric(df, metric_col, title, ylabel, save_name):
    os.makedirs(FIG_DIR, exist_ok=True)

    datasets = ["Set5", "Set14", "BSD100", "Urban100"]
    models = ["Bicubic", "SRCNN", "EDSR-lite", "Attention-EDSR"]

    pivot = df.pivot(index="dataset", columns="model", values=metric_col)
    pivot = pivot.reindex(index=datasets, columns=models)

    ax = pivot.plot(kind="bar", figsize=(10, 6))

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")


def main():
    summary_path = os.path.join(METRIC_DIR, "final_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "Cannot find final_summary.csv. Please run scripts/merge_results.py first."
        )

    df = pd.read_csv(summary_path)

    plot_metric(
        df,
        metric_col="avg_psnr",
        title="PSNR Comparison of Different Super-Resolution Methods",
        ylabel="PSNR (dB)",
        save_name="psnr_comparison.png",
    )

    plot_metric(
        df,
        metric_col="avg_ssim",
        title="SSIM Comparison of Different Super-Resolution Methods",
        ylabel="SSIM",
        save_name="ssim_comparison.png",
    )


if __name__ == "__main__":
    main()