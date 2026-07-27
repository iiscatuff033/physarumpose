import torch.nn as nn


class SkeletonPriorMLP(nn.Module):
    """
    Coordinate-only body-structure prior model.

    Example:
    left_shoulder + left_wrist + task=left_elbow -> left_elbow
    """

    def __init__(self, num_tasks: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        input_dim = 4 + num_tasks
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)
