"""At the exact (oracle) scene_0070 state, where does the specular-supervision error come from?

The specular target of the world loss is I - D, computed from the (clipped) input image and the diffuse
ground truth, whereas the renderer clamps the diffuse and specular buffers separately.  Where the image is
saturated the two disagree.  This script splits the L1 error of the exact state's specular render into
saturated pixels (any channel >= 0.99) and the rest, over the nine training views.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.train import build_pipeline, load_dataset  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    device = torch.device("cuda")
    reference = torch.load("outputs/world_benchmark_v2_scene_0070_strict/last.pt", map_location="cpu", weights_only=False)
    cfg = dict(reference["config"])
    cfg.update(raster_backend="auto", gaussian_init="oracle")
    dataset = load_dataset(cfg)
    train_ids = list(reference["train_ids"])
    pipeline = build_pipeline(dataset, cfg, device, train_ids).eval()
    total = {"sat": [0.0, 0], "nonsat": [0.0, 0]}
    with torch.no_grad():
        for frame_id in train_ids:
            frame = dataset[frame_id]
            image, diffuse = frame.image.to(device), frame.diffuse_gt.to(device)
            render = pipeline(frame.camera.to(device), frame_id, image=image, refine=False)["render"]
            error = (render.specular - (image - diffuse)).abs().mean(0)
            saturated = image.max(0).values >= 0.99
            for key, mask in (("sat", saturated), ("nonsat", ~saturated)):
                total[key][0] += float(error[mask].sum())
                total[key][1] += int(mask.sum())
    n_all = total["sat"][1] + total["nonsat"][1]
    err_all = total["sat"][0] + total["nonsat"][0]
    result = {
        "state": "oracle initialization, nine training views of scene_0070",
        "saturated_pixel_fraction": total["sat"][1] / n_all,
        "specular_l1_all": err_all / n_all,
        "specular_l1_saturated": total["sat"][0] / max(total["sat"][1], 1),
        "specular_l1_non_saturated": total["nonsat"][0] / max(total["nonsat"][1], 1),
        "share_of_error_from_saturated": total["sat"][0] / err_all,
    }
    out = ROOT / "outputs" / "drift_diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "saturation_at_exact_state.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
