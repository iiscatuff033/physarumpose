import torch.nn as nn


class CorrectionHeadMLP(nn.Module):
    """
    Trust-aware correction model.

    Input:
        YOLO keypoints + confidences
        target one-hot
        YOLO target point
        skeleton-prior point
        geometry point
        confidence/anchor features

    Output:
        corrected target x,y in normalized image coordinates
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)
