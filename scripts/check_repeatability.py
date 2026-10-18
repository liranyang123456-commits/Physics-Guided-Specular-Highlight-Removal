"""Repeatability of the experiments: runs repeated with identical commands, and re-evaluation of saved fields.

1. Repeated runs (outputs/v2/repeat/, produced by running the commands of the original runs again with --out outputs/v2/repeat and a
   tag) are compared with the original runs.  The rasterizer accumulates gradients in parallel, so repeated runs are not bit-identical;
   the comparison measures how much they differ with the geometry and lighting fixed and when one of them is estimated.  Earlier copies
   of two runs that were later overwritten (release history of the repository) are included as further repetitions.
2. Saved fields are re-evaluated from their checkpoints (--reevaluate, needs the GPU); the evaluation is a forward pass and must
   reproduce the stored metric files.
Writes outputs/v2/repeatability.json.

    python scripts/check_repeatability.py [--reevaluate]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "v2"
U, REP, FRZ = V2 / "sensitivity_unfrozen", V2 / "repeat", V2 / "sensitivity"
J = lambda p: json.loads(p.read_text(encoding="utf-8"))

GROUPS = {
    "fixed": [   # geometry and lighting fixed
        ("clean, exact priors, seed 0", [FRZ / "clean_seed0.json", REP / "clean_none_seed0_rep1.json", REP / "clean_none_seed0_rep2.json"], []),
        ("reflectance control, scene_0070", [U / "none_none_seed0_mat_control.json", REP / "none_none_seed0_mat_control_rep1.json"], []),
        ("sheen, scene_0070", [U / "none_none_seed0_mat_sheen.json", REP / "none_none_seed0_mat_sheen_rep1.json"], []),
        ("wrapped diffuse, scene_0070", [U / "none_none_seed0_mat_wrap.json", REP / "none_none_seed0_mat_wrap_rep1.json"], []),
        ("anisotropic, extended model, scene_0070", [U / "none_none_seed0_mat_aniso_ext.json", REP / "none_none_seed0_mat_aniso_ext_rep1.json"], []),
    ],
    "estimated": [   # geometry or lighting estimated; earlier copies from the release history (commit 3868421) as extra values
        ("lighting estimated, seed 1", [U / "light_lighting_seed1.json", REP / "light_lighting_seed1_rep1.json", REP / "light_lighting_seed1_rep2.json"],
         [53.8919]),
        ("mild geometry, positions and normals free, seed 1", [U / "geo_mild_geometry_seed1_means_normal.json", REP / "geo_mild_geometry_seed1_rep1.json",
                                                              REP / "geo_mild_geometry_seed1_rep2.json"], [34.3196]),
        ("mild geometry, all views and normal prior, seed 0", [U / "geo_mild_geometry_seed0_imp.json", U / "geo_mild_geometry_seed0_v9_surfAd.json",
                                                              REP / "geo_mild_geometry_seed0_imp_rep1.json"], []),
    ],
}
REEVAL = ["none_none_seed0_mat_control", "s0072_none_none_seed0_mat_oren", "none_none_seed0_mat_wrap_ext", "s0074_none_none_seed0_mat_sheen_ext",
          "geo_mild_geometry_seed0_imp", "s0073_clean_geometry_seed0_imp", "geo_nrm_geometry_seed2_imp", "light_lighting_seed2"]


def reevaluate() -> dict:
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    os.chdir(ROOT)
    from run_sensitivity_unfrozen import install_brdf_extension
    from run_sensitivity_v2 import evaluate
    from mvbrdf_shr.world.train import build_pipeline, load_dataset
    device, out = torch.device("cuda"), {}
    for stem in REEVAL:
        state = torch.load(U / f"{stem}.pt", map_location="cpu", weights_only=False)
        stored = J(U / f"{stem}.json")
        dataset = load_dataset(state["config"])
        pipeline = build_pipeline(dataset, state["config"], device, list(state["train_ids"]))
        keys = state["model"].keys()
        has_d, has_s = "scene.ext_wrap" in keys, "scene.ext_aniso" in keys
        if has_d or has_s:
            install_brdf_extension(pipeline, "both" if has_d and has_s else ("diffuse" if has_d else "specular"))
        pipeline.load_state_dict(state["model"])
        s = evaluate(pipeline, dataset, list(state["holdout_ids"]), device)
        out[stem] = max(abs(s[k]["psnr"] - stored[k]["psnr"]) for k in ("phys", "fuse", "single"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args()
    previous = J(V2 / "repeatability.json") if (V2 / "repeatability.json").exists() else {}
    result = {"groups": {}, "reevaluation": previous.get("reevaluation")}
    for group, items in GROUPS.items():
        rows = []
        for label, files, extra in items:
            vals = [J(f)["phys"]["psnr"] for f in files] + extra
            rows.append({"label": label, "values": vals, "spread": max(vals) - min(vals)})
            print(f"{group:9s} {label:52s} {', '.join(f'{v:.2f}' for v in vals)}  spread {max(vals) - min(vals):.2f} dB")
        result["groups"][group] = {"settings": rows, "max_spread": max(r["spread"] for r in rows), "n_settings": len(rows)}
    if args.reevaluate:
        diffs = reevaluate()
        result["reevaluation"] = {"fields": diffs, "max_difference": max(diffs.values())}
        print(f"re-evaluation of {len(diffs)} saved fields: largest difference {max(diffs.values()):.2e} dB")
    (V2 / "repeatability.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
