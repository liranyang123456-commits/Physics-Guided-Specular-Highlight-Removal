"""Generate per-scene world benchmark configs for the multi-scene holdout study.

Stamps out one YAML per scene from the scene_0070 template, changing only
data_root and output_dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "world" / "benchmark_scene_0070.yaml"
SCENE_GLOB = "data/world_benchmark_v2/scenes/test/scene_{:04d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, nargs="+",
                    default=[70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                    help="Test scene ids (without leading zeros) to generate configs for.")
    ap.add_argument("--config-dir", default=str(ROOT / "configs" / "world"))
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")

    template = TEMPLATE.read_text(encoding="utf-8")
    out_dir = Path(args.config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for scene_id in args.scenes:
        scene_name = f"scene_{scene_id:04d}"
        cfg = template
        # Replace the scene_0070-specific data_root and output_dir.
        cfg = cfg.replace(
            "data/world_benchmark_v2/scenes/test/scene_0070",
            f"data/world_benchmark_v2/scenes/test/{scene_name}",
        )
        cfg = cfg.replace(
            "outputs/world_benchmark_v2_scene_0070_strict",
            f"outputs/world_benchmark_v2_{scene_name}_strict",
        )
        out_path = out_dir / f"benchmark_{scene_name}.yaml"
        out_path.write_text(cfg, encoding="utf-8")
        generated.append(out_path)
        print(f"wrote {out_path.name}  (data_root={scene_name})")
    print(f"\nGenerated {len(generated)} configs. Train each with:")
    print("  python -m mvbrdf_shr.world.train --config-path configs/world "
          f"--config-name benchmark_{scene_name}")
    print("Then evaluate + aggregate:")
    print("  python scripts/eval_world_ablation.py "
          f"--world-checkpoint outputs/world_benchmark_v2_{scene_name}_strict/last.pt "
          f"--output outputs/world_ablation_{scene_name}")
    print("  python scripts/aggregate_world_holdout.py")


if __name__ == "__main__":
    main()
