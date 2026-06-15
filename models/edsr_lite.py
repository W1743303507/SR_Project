import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_features=64, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.residual_scale


class Upsampler(nn.Sequential):
    def __init__(self, scale, num_features):
        if scale < 1 or (scale & (scale - 1)) != 0:
            raise ValueError("Scale must be a positive power of 2")

        layers = []
        for _ in range(int(math.log2(scale))):
            layers.extend(
                [
                    nn.Conv2d(
                        num_features,
                        num_features * 4,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        super().__init__(*layers)


class EDSRLite(nn.Module):
    """Lightweight EDSR that maps an LR RGB image directly to x4 SR."""

    def __init__(
        self,
        scale=4,
        num_features=64,
        num_res_blocks=8,
        residual_scale=0.1,
    ):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[
                ResidualBlock(num_features, residual_scale)
                for _ in range(num_res_blocks)
            ]
        )
        self.body_tail = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1
        )
        self.upsampler = Upsampler(scale, num_features)
        self.tail = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.head(x)
        residual = self.body_tail(self.body(features))
        features = features + residual
        return self.tail(self.upsampler(features))
