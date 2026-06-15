import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(sr_img, hr_img, max_value=255.0):
    """
    Calculate PSNR between SR and HR images.

    Args:
        sr_img: numpy array, RGB image, range 0-255
        hr_img: numpy array, RGB image, range 0-255
    """
    sr_img = sr_img.astype(np.float64)
    hr_img = hr_img.astype(np.float64)

    mse = np.mean((sr_img - hr_img) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def calculate_ssim(sr_img, hr_img):
    """
    Calculate SSIM between SR and HR images.

    Args:
        sr_img: numpy array, RGB image, range 0-255
        hr_img: numpy array, RGB image, range 0-255
    """
    return ssim(
        hr_img,
        sr_img,
        data_range=255,
        channel_axis=2
    )