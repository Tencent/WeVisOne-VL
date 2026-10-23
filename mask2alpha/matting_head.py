from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))


def _resize(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class RefineMattingHead(nn.Module):
    def __init__(
        self,
        feature_channels: Sequence[int] = (32, 64, 256),
        widths: Sequence[int] = (32, 64, 128, 256),
        stem_channels: Sequence[int] = (32, 64),
    ):
        super().__init__()
        f4, f8, f16 = (int(v) for v in feature_channels)
        c2, c4, c8, c16 = (int(v) for v in widths)
        s2, s4 = (int(v) for v in stem_channels)

        self.adapt4 = ConvNormAct(f4, c4, 1)
        self.adapt8 = ConvNormAct(f8, c8, 1)
        self.adapt16 = ConvNormAct(f16, c16, 1)

        self.stem2 = nn.Sequential(ConvNormAct(4, s2, 3, stride=2), ConvNormAct(s2, s2, 3))
        self.stem4 = nn.Sequential(ConvNormAct(s2, s4, 3, stride=2), ConvNormAct(s4, s4, 3))

        self.context = nn.Sequential(
            ConvNormAct(c16 + 1, c16, 3),
            ResidualBlock(c16, dilation=1),
            ResidualBlock(c16, dilation=2),
        )
        self.fuse8 = nn.Sequential(ConvNormAct(c16 + c8 + 1, c8, 3), ConvNormAct(c8, c8, 3))
        self.fuse4 = nn.Sequential(ConvNormAct(c8 + c4 + s4 + 1, c4, 3), ConvNormAct(c4, c4, 3))
        self.fuse2 = nn.Sequential(ConvNormAct(c4 + s2 + 1, c2, 3), ConvNormAct(c2, c2, 3))

        self.out8 = nn.Conv2d(c8, 1, 1)
        self.out4 = nn.Conv2d(c4, 1, 1)
        self.out2 = nn.Conv2d(c2, 1, 1)

    def forward(self, image, prompt, feat4, feat8, feat16):
        full = image.shape[-2:]
        detail2 = self.stem2(torch.cat([image, _resize(prompt, full)], dim=1))
        detail4 = _resize(self.stem4(detail2), feat4.shape[-2:])

        x = self.context(torch.cat([self.adapt16(feat16), _resize(prompt, feat16.shape[-2:])], dim=1))

        x = _resize(x, feat8.shape[-2:])
        x = self.fuse8(torch.cat([x, self.adapt8(feat8), _resize(prompt, feat8.shape[-2:])], dim=1))
        alpha8 = torch.sigmoid(self.out8(x))

        x = _resize(x, feat4.shape[-2:])
        x = self.fuse4(torch.cat([x, self.adapt4(feat4), detail4, _resize(prompt, feat4.shape[-2:])], dim=1))
        alpha4 = torch.sigmoid(self.out4(x))

        x = _resize(x, detail2.shape[-2:])
        x = self.fuse2(torch.cat([x, detail2, _resize(prompt, detail2.shape[-2:])], dim=1))
        alpha2 = torch.sigmoid(self.out2(x))
        return [alpha8, alpha4, alpha2]
