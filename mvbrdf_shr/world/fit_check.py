"""Training-view fit check: use the world-space branch only if the optimized field reproduces its own training images.

The fused output of ``HybridMVBRDFSHR`` trusts the physical branch wherever its full-radiance residual is small.  When the
geometry, lighting, or coverage given to the field are wrong, the field cannot be fitted, but the per-pixel gate does not
detect this reliably on held-out views (see scripts/compare_v2.py and the sensitivity study of the manuscript).  The check
below needs only the training images: it renders the optimized field at the views it was optimized on and compares the full
radiance with the observed images.  Fields that fail (PSNR below ``tau``) are rejected and the single-image prediction is
returned for the whole scene.
"""
from __future__ import annotations

import numpy as np
import torch

from ..metrics import psnr_torch

DEFAULT_TAU_DB = 40.0


@torch.no_grad()
def training_fit_psnr(pipeline, dataset, train_ids, device) -> float:
    """Mean full-radiance PSNR (dB) of the field on the views it was optimized on."""
    pipeline.eval()
    values = []
    for frame_id in train_ids:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
        full = (render.diffuse + render.specular).clamp(0, 1)
        values.append(float(psnr_torch(full.unsqueeze(0), image.unsqueeze(0))))
    return float(np.mean(values))


def accept_field(fit_psnr: float, tau: float = DEFAULT_TAU_DB) -> bool:
    return fit_psnr >= tau


def fuse_with_check(hybrid_pred: torch.Tensor, single_pred: torch.Tensor, fit_psnr: float, tau: float = DEFAULT_TAU_DB) -> torch.Tensor:
    """Hybrid output if the field passes the check, otherwise the single-image prediction."""
    return hybrid_pred if accept_field(fit_psnr, tau) else single_pred
