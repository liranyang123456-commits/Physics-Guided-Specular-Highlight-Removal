"""Train the world field of the corrected recipe (configs/world/v2) for several scenes with bounded concurrency.

  python scripts/make_v2_configs.py --steps 1500          # writes configs/world/v2/*.yaml
  python scripts/run_v2_world.py --scenes 70 71 72 ...    # trains outputs/v2/world_scene_00NN/last.pt
  python scripts/run_v2_world.py --scenes 70 --extra-runs 2   # outputs/v2/world_scene_0070_run1, _run2 (independent runs)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(70, 80)))
    parser.add_argument("--extra-runs", type=int, default=0, help="additional independent runs of the first scene")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config-dir", default="configs/world/v2")
    args = parser.parse_args()

    jobs = []
    for scene in args.scenes:
        jobs.append((f"scene_00{scene}", ROOT / args.config_dir / f"benchmark_scene_00{scene}_v2.yaml"))
    tmp = ROOT / "outputs" / "_tmp_configs"
    tmp.mkdir(parents=True, exist_ok=True)
    for k in range(1, args.extra_runs + 1):
        scene = args.scenes[0]
        cfg = yaml.safe_load((ROOT / args.config_dir / f"benchmark_scene_00{scene}_v2.yaml").read_text(encoding="utf-8"))
        cfg["output_dir"] = cfg["output_dir"] + f"_run{k}"
        path = tmp / f"scene_00{scene}_run{k}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        jobs.append((f"scene_00{scene}_run{k}", path))

    running, t0 = [], time.time()
    (ROOT / "outputs" / "v2").mkdir(parents=True, exist_ok=True)
    while jobs or running:
        while jobs and len(running) < args.workers:
            name, cfg_path = jobs.pop(0)
            log = open(ROOT / "outputs" / "v2" / f"train_{name}.log", "w", encoding="utf-8")
            running.append((name, subprocess.Popen(f'python -m mvbrdf_shr.world.train --config "{cfg_path}"', cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT, shell=True)))
        for name, proc in list(running):
            if proc.poll() is not None:
                running.remove((name, proc))
                print(f"[{time.time() - t0:5.0f}s] finished {name} (exit {proc.returncode}); {len(jobs)} queued", flush=True)
        time.sleep(3)
    print("all world trainings finished", flush=True)


if __name__ == "__main__":
    main()
