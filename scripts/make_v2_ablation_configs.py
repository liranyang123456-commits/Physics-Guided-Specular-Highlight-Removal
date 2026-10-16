"""Configs for the two ablation rows that need a dedicated training run under the corrected recipe (revision, step A).

* strict_nossim : configs/world/benchmark_scene_0070.yaml (oracle geometry + albedo initialisation, paired diffuse/specular
                  supervision, learned world refiner from step 1500) with ONLY the 7x7 SSIM photometric term removed.
                  Like-for-like counterpart of the original strict-holdout rows (physical diffuse and world refiner).
* bootstrap     : configs/world/v2/benchmark_scene_0070_v2.yaml (albedo-free, RGB-only) with the albedo initialised from the
                  single-image pseudo-diffuse prediction (counterpart of the 'inverse + single-image bootstrap' row).

    python scripts/make_v2_ablation_configs.py
    python -m mvbrdf_shr.world.train --config configs/world/v2_ablation/strict_nossim.yaml
    python -m mvbrdf_shr.world.train --config configs/world/v2_ablation/bootstrap.yaml
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "world" / "v2_ablation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    strict = yaml.safe_load((ROOT / "configs/world/benchmark_scene_0070.yaml").read_text(encoding="utf-8"))
    strict["loss"]["photo_ssim"] = 0.0
    strict["output_dir"] = "outputs/v2/ablation_strict_nossim"
    (OUT / "strict_nossim.yaml").write_text(yaml.safe_dump(strict, sort_keys=False), encoding="utf-8")

    boot = yaml.safe_load((ROOT / "configs/world/v2/benchmark_scene_0070_v2.yaml").read_text(encoding="utf-8"))
    pro = yaml.safe_load((ROOT / "configs/world/benchmark_scene_0070_inverse_pro.yaml").read_text(encoding="utf-8"))
    for key in ("pseudo_diffuse_checkpoint", "pseudo_diffuse_blend", "pseudo_diffuse_confidence_floor"):
        boot[key] = pro[key]
    boot["output_dir"] = "outputs/v2/ablation_bootstrap"
    (OUT / "bootstrap.yaml").write_text(yaml.safe_dump(boot, sort_keys=False), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
