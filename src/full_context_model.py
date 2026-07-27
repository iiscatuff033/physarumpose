import torch.nn as nn


class FullSkeletonPriorMLP(nn.Module):
    """
    Full-body context missing-joint recovery model.

    Input:
        16 joints * (x, y, mask) + target_one_hot

    Output:
        target joint x,y
    """

    def __init__(self, num_joints: int = 16, num_targets: int = 4, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()

        input_dim = num_joints * 3 + num_targets

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
