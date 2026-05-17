"""Run a trained sparse-reward model on a single (image, state) example.

Useful for live monitoring next to ACT inference: every tick, feed the
same observation into this model to get P(task done) and surface a
"done" signal once the running average crosses a threshold.

Example
-------
>>> from inference import RewardScorer
>>> scorer = RewardScorer.from_checkpoint("outputs/run/best.pt")
>>> prob = scorer.score(frame_rgb_uint8, np.array([...7 floats...]))
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import SparseRewardConfig, build_eval_transform  # noqa: E402
from reward_model import RewardModelConfig, SparseRewardModel  # noqa: E402


@dataclass
class RewardScorer:
    model: SparseRewardModel
    transform: T.Compose
    device: str
    image_size: int

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> "RewardScorer":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        model_cfg = RewardModelConfig(**ckpt["model_cfg"])
        ds_cfg = SparseRewardConfig(**ckpt["ds_cfg"])
        model = SparseRewardModel(model_cfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        transform = build_eval_transform(ds_cfg)
        return cls(model=model, transform=transform, device=device, image_size=ds_cfg.image_size)

    @torch.no_grad()
    def score(self, image, state) -> float:
        """image: HxWx3 uint8 RGB or torch CHW tensor; state: 1D iterable
        of `state_dim` floats. Returns P(done)."""
        img_t = _coerce_image(image)
        img_t = self.transform(img_t).unsqueeze(0).to(self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
        logit = self.model(img_t, state_t)
        return float(torch.sigmoid(logit).item())

    @torch.no_grad()
    def score_batch(self, images, states) -> np.ndarray:
        img_t = torch.stack([self.transform(_coerce_image(i)) for i in images]).to(self.device)
        state_t = torch.as_tensor(np.asarray(states), dtype=torch.float32).to(self.device)
        logits = self.model(img_t, state_t)
        return torch.sigmoid(logits).cpu().numpy()


def _coerce_image(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        t = image
    else:
        t = torch.from_numpy(np.ascontiguousarray(image))
    if t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    return t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--image", required=True, help="Path to RGB image (jpg/png).")
    parser.add_argument(
        "--state",
        required=True,
        help="Comma-separated state floats matching the trained state_dim.",
    )
    args = parser.parse_args()

    from PIL import Image

    img = np.array(Image.open(args.image).convert("RGB"))
    state = np.array([float(x) for x in args.state.split(",")], dtype=np.float32)

    scorer = RewardScorer.from_checkpoint(args.checkpoint)
    prob = scorer.score(img, state)
    print(f"P(task done) = {prob:.4f}")
