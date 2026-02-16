from __future__ import annotations

import torch
import torch.nn as nn


class FrozenBackbone(nn.Module):
    """A tiny frozen backbone placeholder for multimodal reward modeling.

    This keeps v0.3.2 lightweight and open-source friendly: users can swap this
    module with a real VLM encoder while keeping the RM head and pairwise loss.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MinimalVLMRewardModel(nn.Module):
    """Freeze backbone features and train only a scalar reward head."""

    def __init__(self, text_dim: int = 256, image_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.text_backbone = FrozenBackbone(text_dim, hidden_dim)
        self.image_backbone = FrozenBackbone(image_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_backbone(text_feat)
        i = self.image_backbone(image_feat)
        fused = torch.cat([t, i], dim=-1)
        return self.head(fused).squeeze(-1)
