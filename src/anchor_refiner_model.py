import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceAnchorRefiner(nn.Module):
    """
    Crop-based heatmap model.

    Input:
        target-person/path crop

    Output channels:
        0: anchor A heatmap
        1: anchor B heatmap
        2: hidden target-region heatmap
    """
    def __init__(self, out_channels=3, base=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 128

            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 64

            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, out_channels, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


def soft_argmax_2d(logits, temperature=0.05):
    """
    logits: [B,C,H,W]
    returns xy [B,C,2] normalized crop coordinates
    """
    b, c, h, w = logits.shape
    flat = logits.reshape(b, c, -1)
    prob = F.softmax(flat / max(float(temperature), 1e-6), dim=-1)

    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    x = torch.sum(prob * xx.view(1, 1, -1), dim=-1)
    y = torch.sum(prob * yy.view(1, 1, -1), dim=-1)
    xy = torch.stack([x, y], dim=-1)

    conf = torch.sigmoid(logits).reshape(b, c, -1).amax(dim=-1)
    return xy, conf
