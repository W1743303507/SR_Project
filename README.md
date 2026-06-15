# 基于深度学习的图像超分辨率重建

本项目实现了基于深度学习的单幅图像超分辨率重建系统，包含 Bicubic、SRCNN、EDSR-lite 和 Attention-EDSR 等方法。

## 已实现功能

- 数据集预处理与 LR/HR 图像对生成
- PyTorch Dataset/DataLoader
- Bicubic baseline
- SRCNN 训练与测试
- EDSR-lite 训练与测试
- Attention-EDSR 改进模型
- PSNR / SSIM / 推理时间评估

## 数据集

使用 DIV2K、Flickr2K、Set5、Set14、BSD100、Urban100。

由于数据集文件较大，`data/raw`、`data/processed` 和模型权重文件不会上传至 GitHub。

## 运行示例

```bash
D:\Miniconda\envs\sr_project\python.exe scripts\prepare_dataset.py --scale 4
D:\Miniconda\envs\sr_project\python.exe scripts\run_bicubic_baseline.py
D:\Miniconda\envs\sr_project\python.exe scripts\train_srcnn.py
D:\Miniconda\envs\sr_project\python.exe scripts\test_srcnn.py
D:\Miniconda\envs\sr_project\python.exe scripts\train_edsr.py --epochs 30
D:\Miniconda\envs\sr_project\python.exe scripts\test_edsr.py