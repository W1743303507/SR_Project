import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN for x4 image super-resolution.

    Input:
        LR image tensor [B, 3, H, W]

    Output:
        SR image tensor [B, 3, H*scale, W*scale]
    """

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First enlarge LR image to HR size using bicubic interpolation
        x = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # Keep pixel range between 0 and 1
        x = torch.clamp(x, 0.0, 1.0)

        return x