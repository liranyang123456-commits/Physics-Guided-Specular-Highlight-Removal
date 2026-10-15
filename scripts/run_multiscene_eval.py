"""Re-evaluate the ten strict-holdout scenes with each scene's training rasterizer.

The backend per scene comes from outputs/world_raster_backend_check.json
(scripts/check_raster_backend.py); results go to outputs/world_ablation_scene_00NN.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    backends = json.loads((ROOT / "outputs" / "world_raster_backend_check.json").read_text())
    for scene in range(70, 80):
        backend = backends[str(scene)]["training_backend"]
        cmd = [
            sys.executable, "scripts/eval_world_ablation.py",
            "--world-checkpoint", f"outputs/world_benchmark_v2_scene_00{scene}_strict/last.pt",
            "--output", f"outputs/world_ablation_scene_00{scene}",
            "--raster-backend", backend,
        ]
        print(f"scene_00{scene} ({backend})", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
