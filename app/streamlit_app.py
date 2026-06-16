from pathlib import Path
import io
import sys
import time

import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.srcnn import SRCNN
from models.edsr_lite import EDSRLite
from models.attention_edsr import AttentionEDSR


SCALE = 4

CHECKPOINTS = {
    "SRCNN": PROJECT_ROOT / "results" / "checkpoints" / "srcnn_x4_best.pth",
    "EDSR-lite": PROJECT_ROOT / "results" / "checkpoints" / "edsr_lite_x4_best.pth",
    "Attention-EDSR": PROJECT_ROOT / "results" / "checkpoints" / "attention_edsr_x4_best.pth",
}

FINAL_SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "final_summary.csv"


FALLBACK_RESULTS = [
    {"model": "Bicubic", "dataset": "Set5", "avg_psnr": 26.6477, "avg_ssim": 0.7902, "avg_time": 0.000793},
    {"model": "Bicubic", "dataset": "Set14", "avg_psnr": 24.2089, "avg_ssim": 0.6856, "avg_time": 0.001277},
    {"model": "Bicubic", "dataset": "BSD100", "avg_psnr": 24.6508, "avg_ssim": 0.6614, "avg_time": 0.000979},
    {"model": "Bicubic", "dataset": "Urban100", "avg_psnr": 21.6991, "avg_ssim": 0.6517, "avg_time": 0.005613},

    {"model": "SRCNN", "dataset": "Set5", "avg_psnr": 27.2939, "avg_ssim": 0.8073, "avg_time": 0.052361},
    {"model": "SRCNN", "dataset": "Set14", "avg_psnr": 24.6825, "avg_ssim": 0.7046, "avg_time": 0.021852},
    {"model": "SRCNN", "dataset": "BSD100", "avg_psnr": 25.0454, "avg_ssim": 0.6849, "avg_time": 0.026534},
    {"model": "SRCNN", "dataset": "Urban100", "avg_psnr": 22.1369, "avg_ssim": 0.6755, "avg_time": 0.074522},

    {"model": "EDSR-lite", "dataset": "Set5", "avg_psnr": 28.1507, "avg_ssim": 0.8346, "avg_time": 0.076366},
    {"model": "EDSR-lite", "dataset": "Set14", "avg_psnr": 25.1991, "avg_ssim": 0.7290, "avg_time": 0.011899},
    {"model": "EDSR-lite", "dataset": "BSD100", "avg_psnr": 25.4220, "avg_ssim": 0.7054, "avg_time": 0.035999},
    {"model": "EDSR-lite", "dataset": "Urban100", "avg_psnr": 22.6504, "avg_ssim": 0.7021, "avg_time": 2.966157},

    {"model": "Attention-EDSR", "dataset": "Set5", "avg_psnr": 28.3941, "avg_ssim": 0.8417, "avg_time": 0.081463},
    {"model": "Attention-EDSR", "dataset": "Set14", "avg_psnr": 25.3613, "avg_ssim": 0.7339, "avg_time": 0.017757},
    {"model": "Attention-EDSR", "dataset": "BSD100", "avg_psnr": 25.5019, "avg_ssim": 0.7088, "avg_time": 0.050099},
    {"model": "Attention-EDSR", "dataset": "Urban100", "avg_psnr": 22.7903, "avg_ssim": 0.7100, "avg_time": 0.102903},
]


MODEL_DESCRIPTIONS = {
    "Bicubic": "传统双三次插值方法，不需要训练，用作基础对照方法。",
    "SRCNN": "浅层卷积神经网络模型，用于验证深度学习方法相较传统插值的提升。",
    "EDSR-lite": "轻量化残差超分辨率模型，使用残差结构和 PixelShuffle 完成 x4 重建。",
    "Attention-EDSR": "本项目改进模型，在 EDSR-lite 基础上加入通道注意力机制，进一步增强重要特征表达。",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_state_dict(value):
    return isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values())


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)
        cleaned[key] = value
    return cleaned


def extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if hasattr(value, "state_dict") and not isinstance(value, dict):
            return value.state_dict()
        if is_state_dict(value):
            return value

    if is_state_dict(checkpoint):
        return checkpoint

    raise KeyError("No valid model state_dict was found in the checkpoint.")


def checkpoint_config(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}

    return {
        "scale": checkpoint.get("scale", SCALE),
        "num_features": checkpoint.get("num_features", 64),
        "num_res_blocks": checkpoint.get("num_res_blocks", 8),
        "residual_scale": checkpoint.get("residual_scale", 0.1),
        "reduction": checkpoint.get("reduction", 16),
    }


def build_model(method, checkpoint=None):
    cfg = checkpoint_config(checkpoint)

    if method == "SRCNN":
        return SRCNN(scale=cfg.get("scale", SCALE))

    if method == "EDSR-lite":
        return EDSRLite(
            scale=cfg.get("scale", SCALE),
            num_features=cfg.get("num_features", 64),
            num_res_blocks=cfg.get("num_res_blocks", 8),
            residual_scale=cfg.get("residual_scale", 0.1),
        )

    if method == "Attention-EDSR":
        return AttentionEDSR(
            scale=cfg.get("scale", SCALE),
            num_features=cfg.get("num_features", 64),
            num_res_blocks=cfg.get("num_res_blocks", 8),
            residual_scale=cfg.get("residual_scale", 0.1),
            reduction=cfg.get("reduction", 16),
        )

    raise ValueError(f"Unknown method: {method}")


@st.cache_resource(show_spinner=False)
def load_model(method):
    device = get_device()
    checkpoint_path = CHECKPOINTS[method]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(method, checkpoint).to(device)

    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, device


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)

    if array.shape[-1] == 4:
        array = array[..., :3]

    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_image(tensor):
    array = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def run_bicubic(image):
    start = time.perf_counter()
    result = image.resize(
        (image.width * SCALE, image.height * SCALE),
        Image.BICUBIC,
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_model(image, method):
    model, device = load_model(method)
    lr_tensor = image_to_tensor(image).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    with torch.inference_mode():
        sr_tensor = model(lr_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    return tensor_to_image(sr_tensor), elapsed


def run_super_resolution(image, method):
    if method == "Bicubic":
        return run_bicubic(image)
    return run_model(image, method)


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def load_summary_dataframe():
    if FINAL_SUMMARY_PATH.exists():
        try:
            df = pd.read_csv(FINAL_SUMMARY_PATH)
            required = {"model", "dataset", "avg_psnr", "avg_ssim"}
            if required.issubset(set(df.columns)):
                return df, True
        except Exception:
            pass

    return pd.DataFrame(FALLBACK_RESULTS), False


def show_metric_cards(summary):
    attention = summary[summary["model"] == "Attention-EDSR"]
    edsr = summary[summary["model"] == "EDSR-lite"]

    if attention.empty or edsr.empty:
        return

    avg_attention_psnr = attention["avg_psnr"].mean()
    avg_attention_ssim = attention["avg_ssim"].mean()
    avg_edsr_psnr = edsr["avg_psnr"].mean()
    avg_edsr_ssim = edsr["avg_ssim"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("最佳模型", "Attention-EDSR")
    col2.metric("平均 PSNR", f"{avg_attention_psnr:.4f} dB", f"{avg_attention_psnr - avg_edsr_psnr:+.4f}")
    col3.metric("平均 SSIM", f"{avg_attention_ssim:.4f}", f"{avg_attention_ssim - avg_edsr_ssim:+.4f}")
    col4.metric("放大倍率", "x4")


def show_results_summary():
    st.subheader("实验结果汇总")

    summary, from_file = load_summary_dataframe()

    if from_file:
        st.success(f"已读取实验结果文件：{FINAL_SUMMARY_PATH}")
    else:
        st.warning("未找到 results/metrics/final_summary.csv，当前页面使用内置最终实验结果展示。")

    show_metric_cards(summary)

    st.markdown("### 完整结果表")
    display_df = summary.copy()
    display_df["avg_psnr"] = display_df["avg_psnr"].round(4)
    display_df["avg_ssim"] = display_df["avg_ssim"].round(4)
    if "avg_time" in display_df.columns:
        display_df["avg_time"] = display_df["avg_time"].round(6)

    st.dataframe(display_df, use_container_width=True)

    datasets_order = ["Set5", "Set14", "BSD100", "Urban100"]
    models_order = ["Bicubic", "SRCNN", "EDSR-lite", "Attention-EDSR"]

    psnr_table = summary.pivot(index="dataset", columns="model", values="avg_psnr")
    ssim_table = summary.pivot(index="dataset", columns="model", values="avg_ssim")

    psnr_table = psnr_table.reindex(index=datasets_order, columns=models_order)
    ssim_table = ssim_table.reindex(index=datasets_order, columns=models_order)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PSNR 对比")
        st.bar_chart(psnr_table)

    with col2:
        st.markdown("### SSIM 对比")
        st.bar_chart(ssim_table)

    st.markdown("### Attention-EDSR 相比 EDSR-lite 的提升")

    edsr = summary[summary["model"] == "EDSR-lite"].set_index("dataset")
    attention = summary[summary["model"] == "Attention-EDSR"].set_index("dataset")

    rows = []
    for dataset in datasets_order:
        if dataset in edsr.index and dataset in attention.index:
            rows.append(
                {
                    "dataset": dataset,
                    "psnr_gain": attention.loc[dataset, "avg_psnr"] - edsr.loc[dataset, "avg_psnr"],
                    "ssim_gain": attention.loc[dataset, "avg_ssim"] - edsr.loc[dataset, "avg_ssim"],
                }
            )

    improve_df = pd.DataFrame(rows)
    if not improve_df.empty:
        avg_row = pd.DataFrame(
            [
                {
                    "dataset": "Average",
                    "psnr_gain": improve_df["psnr_gain"].mean(),
                    "ssim_gain": improve_df["ssim_gain"].mean(),
                }
            ]
        )
        improve_df = pd.concat([improve_df, avg_row], ignore_index=True)
        improve_df["psnr_gain"] = improve_df["psnr_gain"].round(4)
        improve_df["ssim_gain"] = improve_df["ssim_gain"].round(4)
        st.dataframe(improve_df, use_container_width=True)

    st.info(
        "实验结果表明，Attention-EDSR 在 Set5、Set14、BSD100 和 Urban100 四个测试集上均取得最高 PSNR 和 SSIM，"
        "说明通道注意力机制能够进一步提升超分辨率重建质量。"
    )


def show_single_model_demo(uploaded_file, method):
    lr_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner(f"正在使用 {method} 进行 x4 超分辨率重建..."):
        sr_image, elapsed = run_super_resolution(lr_image, method)

    col1, col2, col3 = st.columns(3)
    col1.metric("输入尺寸", f"{lr_image.width} × {lr_image.height}")
    col2.metric("输出尺寸", f"{sr_image.width} × {sr_image.height}")
    col3.metric("推理时间", f"{elapsed:.4f} s")

    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.markdown("### 输入低分辨率图像")
        st.image(lr_image, use_container_width=True)

    with image_col2:
        st.markdown(f"### {method} 重建结果")
        st.image(sr_image, use_container_width=True)

    st.download_button(
        label="下载重建结果",
        data=image_to_png_bytes(sr_image),
        file_name=f"{method.lower().replace('-', '_')}_x4_sr.png",
        mime="image/png",
    )


def show_all_models_demo(uploaded_file):
    lr_image = Image.open(uploaded_file).convert("RGB")

    methods = ["Bicubic", "SRCNN", "EDSR-lite", "Attention-EDSR"]
    results = {}

    progress = st.progress(0)
    status = st.empty()

    for index, method in enumerate(methods):
        status.write(f"正在运行：{method}")
        sr_image, elapsed = run_super_resolution(lr_image, method)
        results[method] = {"image": sr_image, "time": elapsed}
        progress.progress((index + 1) / len(methods))

    status.success("四种方法对比完成。")

    st.markdown("### 输入图像")
    st.image(lr_image, use_container_width=False, caption=f"LR 输入：{lr_image.width} × {lr_image.height}")

    st.markdown("### 四模型重建效果对比")

    cols = st.columns(4)
    for col, method in zip(cols, methods):
        with col:
            st.markdown(f"#### {method}")
            st.image(results[method]["image"], use_container_width=True)
            st.caption(f"推理时间：{results[method]['time']:.4f} s")

    time_df = pd.DataFrame(
        [
            {"method": method, "inference_time": results[method]["time"]}
            for method in methods
        ]
    )

    st.markdown("### 推理时间对比")
    st.dataframe(time_df, use_container_width=True)
    st.bar_chart(time_df.set_index("method"))

    best_image = results["Attention-EDSR"]["image"]
    st.download_button(
        label="下载 Attention-EDSR 最优结果",
        data=image_to_png_bytes(best_image),
        file_name="attention_edsr_x4_sr.png",
        mime="image/png",
    )


def show_project_description():
    st.subheader("项目说明")

    st.markdown(
        """
        本系统实现了单幅图像超分辨率重建任务，目标是将低分辨率图像恢复为更高分辨率图像。
        项目采用 x4 放大倍率，并完成了传统插值方法与深度学习模型的对比实验。
        """
    )

    st.markdown("### 技术路线")

    st.markdown(
        """
        1. **Bicubic**：传统双三次插值方法，作为基础对照。
        2. **SRCNN**：浅层卷积神经网络，验证深度学习方法的有效性。
        3. **EDSR-lite**：基于残差结构的轻量化超分辨率模型。
        4. **Attention-EDSR**：在 EDSR-lite 中加入通道注意力机制，是本项目的改进模型。
        """
    )

    st.markdown("### 评价指标")

    st.markdown(
        """
        - **PSNR**：峰值信噪比，数值越高表示像素误差越小。
        - **SSIM**：结构相似性，数值越高表示图像结构越接近真实高分辨率图像。
        """
    )

    st.markdown("### 实验结论")

    st.success(
        "最终实验结果表明：Bicubic < SRCNN < EDSR-lite < Attention-EDSR。"
        "改进模型 Attention-EDSR 在四个测试集上均取得最优结果。"
    )


def main():
    st.set_page_config(
        page_title="基于深度学习的图像超分辨率重建系统",
        page_icon="🖼️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .sub-title {
            font-size: 18px;
            color: #666666;
            margin-bottom: 24px;
        }
        .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">基于深度学习的图像超分辨率重建系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Image Super-Resolution Reconstruction Based on Deep Learning, x4</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("系统设置")

        mode = st.radio(
            "展示模式",
            ["单模型重建", "四模型对比"],
        )

        method = st.selectbox(
            "选择模型",
            ["Bicubic", "SRCNN", "EDSR-lite", "Attention-EDSR"],
            index=3,
            disabled=(mode == "四模型对比"),
        )

        st.markdown("---")
        st.write(f"当前设备：`{get_device()}`")
        st.write(f"放大倍率：`x{SCALE}`")

        st.markdown("---")
        st.subheader("模型说明")
        if mode == "单模型重建":
            st.write(MODEL_DESCRIPTIONS[method])
        else:
            st.write("四模型对比模式会依次运行 Bicubic、SRCNN、EDSR-lite 和 Attention-EDSR，并展示重建效果。")

    tab_demo, tab_results, tab_about = st.tabs(["系统演示", "实验结果", "项目说明"])

    with tab_demo:
        st.subheader("图像超分辨率重建演示")

        uploaded_file = st.file_uploader(
            "上传一张低分辨率图片",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        if uploaded_file is None:
            st.info("请上传一张低分辨率图像，系统会输出 x4 超分辨率重建结果。")
        else:
            if mode == "单模型重建":
                show_single_model_demo(uploaded_file, method)
            else:
                show_all_models_demo(uploaded_file)

    with tab_results:
        show_results_summary()

    with tab_about:
        show_project_description()


if __name__ == "__main__":
    main()

