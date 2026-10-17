"""Share of object pixels with a saturated channel (8-bit value 255) in the inputs and the diffuse targets of the twenty test
scenes of world_benchmark_v2 (all twelve views).  Writes outputs/v2/benchmark_saturation.json.

    python scripts/benchmark_saturation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    per = {}
    for n in range(70, 90):
        d = ROOT / f"data/world_benchmark_v2/scenes/test/scene_00{n}"
        inp, dif = [], []
        for v in range(12):
            x = np.asarray(Image.open(d / f"images/{v:03d}.png"))
            y = np.asarray(Image.open(d / f"diffuse/{v:03d}.png"))
            obj = y.max(-1) > 5
            inp.append((x[obj].max(-1) == 255).mean())
            dif.append((y[obj].max(-1) == 255).mean())
        per[f"scene_00{n}"] = {"input": float(np.mean(inp)), "diffuse": float(np.mean(dif))}
    vals_i = np.array([v["input"] for v in per.values()])
    vals_d = np.array([v["diffuse"] for v in per.values()])
    out = {"per_scene": per, "input_mean": float(vals_i.mean()), "diffuse_mean": float(vals_d.mean()),
           "diffuse_min": float(vals_d.min()), "diffuse_max": float(vals_d.max())}
    (ROOT / "outputs/v2/benchmark_saturation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_scene"}, indent=2))


if __name__ == "__main__":
    main()
