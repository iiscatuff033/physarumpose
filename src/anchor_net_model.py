import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AnchorNet(nn.Module):
    """
    Lightweight heatmap model for visible anchors only.

    Input:  RGB crop [B,3,256,256]
    Output: heatmaps [B,K,64,64], visibility logits [B,K]
    """

    def __init__(self, num_anchors=8, base=32):
        super().__init__()
        self.num_anchors = num_anchors

        self.stem = ConvBlock(3, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base, base * 2))      # 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base * 2, base * 4))  # 64
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base * 4, base * 8))  # 32

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base * 8, base * 4),
        )  # 64

        self.heatmap_head = nn.Conv2d(base * 4, num_anchors, kernel_size=1)

        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base * 4, num_anchors),
        )

    def forward(self, x):
        s = self.stem(x)
        d1 = self.down1(s)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        up = self.up1(d3)
        heatmaps = self.heatmap_head(up)
        vis_logits = self.vis_head(d3)
        return heatmaps, vis_logits
