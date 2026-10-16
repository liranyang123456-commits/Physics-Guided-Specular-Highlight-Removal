"""Evaluate the v2 world fields with scripts/eval_world_ablation.py (same protocol as the original ten-scene study).

Writes outputs/v2/ablation_scene_00NN/ablation_metrics.json and, for the extra runs of scene_0070,
outputs/v2/ablation_scene_0070_run{k}/ablation_metrics.json.  The gate is the one trained on the v2 scene_0070 field.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--extra-runs", type=int, default=2)
    args = parser.parse_args()
    names = [f"scene_00{n}" for n in range(70, 80)] + [f"scene_0070_run{k}" for k in range(1, args.extra_runs + 1)]
    jobs = [(n, [sys.executable, "scripts/eval_world_ablation.py", "--world-checkpoint", f"outputs/v2/world_{n}/last.pt",
                 "--fusion-checkpoint", "outputs/v2/fusion_scene_0070/fusion.pt", "--output", f"outputs/v2/ablation_{n}",
                 "--raster-backend", "gsplat"]) for n in names]
    running, t0 = [], time.time()
    while jobs or running:
        while jobs and len(running) < args.workers:
            n, cmd = jobs.pop(0)
            running.append((n, subprocess.Popen(cmd, cwd=ROOT, stdout=open(ROOT / "outputs" / "v2" / f"eval_{n}.log", "w"),
                                                stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(ROOT)))))
        for n, proc in list(running):
            if proc.poll() is not None:
                running.remove((n, proc))
                print(f"[{time.time() - t0:4.0f}s] evaluated {n} (exit {proc.returncode})", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
