"""Agreement of the two renderings of each mismatch scene (scripts/make_mismatch_scenes.py): PSNR between the Mitsuba and our
diffuse images and input images, over the whole image and over the object interior (3 px inside both silhouettes), for the
materials of the original Gaussians and for the smoothed materials.  Writes outputs/v2/renderer_agreement.json.

    python scripts/compare_renderers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "world_benchmark_mismatch"
PAIRS = {"original": ("mitsuba", "ours"), "smooth": ("mitsuba_smooth", "ours_smooth")}


def load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p)).astype(np.float32) / 255


def main() -> None:
    out = {}
    for name, (m_dir, o_dir) in PAIRS.items():
        res = {"diffuse_full": [], "diffuse_interior": [], "input_full": [], "input_interior": []}
        for n in range(70, 80):
            for v in range(12):
                for kind, key in (("diffuse", "diffuse"), ("images", "input")):
                    a = load(D / m_dir / "scenes/test" / f"scene_00{n}" / kind / f"{v:03d}.png")
                    b = load(D / o_dir / "scenes/test" / f"scene_00{n}" / kind / f"{v:03d}.png")
                    inner = binary_erosion((a.max(-1) > 0.02) & (b.max(-1) > 0.02), iterations=3)
                    res[f"{key}_full"].append(-10 * np.log10(((a - b) ** 2).mean()))
                    res[f"{key}_interior"].append(-10 * np.log10(((a - b) ** 2)[inner].mean()))
        out[name] = {k: float(np.mean(v)) for k, v in res.items()}
        print(name, {k: round(v, 2) for k, v in out[name].items()})
    (ROOT / "outputs/v2/renderer_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
