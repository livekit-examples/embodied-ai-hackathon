"""Sparse reward model.

Takes the same single-step inputs as ACT (image + 7-dim proprio state, no
temporal buffer) and outputs a single logit. Sigmoid(logit) ≈ P(task is in
its final ~0.5s window).

Encoder: ResNet18 backbone (ImageNet-init) → 512-d vec; state MLP 7→64;
fused MLP → 1 logit. Light, fits comfortably in a few hundred MB, trains
in minutes on a single GPU for the typical ~50-episode dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class RewardModelConfig:
    state_dim: int = 7
    image_size: int = 224
    hidden_dim: int = 256
    state_hidden: int = 64
    dropout: float = 0.1
    pretrained_backbone: bool = True
    freeze_backbone_stem: bool = True  # freeze conv1+bn1+layer1, lets small datasets generalise

    # Normalization params for ImageNet-pretrained ResNet. Input frames are
    # expected as float in [0, 1], CHW.
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)


class SparseRewardModel(nn.Module):
    def __init__(self, cfg: RewardModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RewardModelConfig()

        weights = ResNet18_Weights.IMAGENET1K_V1 if self.cfg.pretrained_backbone else None
        backbone = resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if self.cfg.freeze_backbone_stem:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1")):
                    p.requires_grad_(False)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.cfg.state_dim, self.cfg.state_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.state_hidden, self.cfg.state_hidden),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + self.cfg.state_hidden, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, 1),
        )

        mean = torch.tensor(self.cfg.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.cfg.image_std).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.image_mean) / self.image_std

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) float in [0, 1]; state: (B, state_dim).
        returns: (B,) logits.
        """
        x = self.normalize_image(image)
        img_feat = self.backbone(x)
        state_feat = self.state_encoder(state)
        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.head(fused).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(image, state))
