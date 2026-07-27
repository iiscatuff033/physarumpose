import torch
import torch.nn as nn


class ResidualGateMLP(nn.Module):
    """
    Model outputs:
    - residual dx, dy from YOLO point
    - gate logit: should correct or keep YOLO?
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.20):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        self.residual_head = nn.Linear(hidden_dim // 2, 2)
        self.gate_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.backbone(x)
        residual = self.residual_head(h)
        gate_logit = self.gate_head(h).squeeze(-1)
        return residual, gate_logit
