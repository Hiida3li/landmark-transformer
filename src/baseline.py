"""MLP baseline: mean-pool over time, then a 2-layer MLP. No attention, no time."""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, n_features: int = 63, n_classes: int = 5,
                 hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 63) -> mean over time -> (B, 63) -> logits (B, n_classes)
        return self.net(x.mean(dim=1))


if __name__ == "__main__":
    m = MLPBaseline()
    print("parameters:", f"{sum(p.numel() for p in m.parameters()):,}")
    print(m(torch.randn(4, 30, 63)).shape)