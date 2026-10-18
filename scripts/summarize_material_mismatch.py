"""Summary of the material-model mismatch study (scenes of scripts/make_material_mismatch_scenes.py).

For every variant: how far the input images and the diffuse targets differ from the control, the held-out diffuse PSNR of the
physical branch, the fusion, and the single-image branch with the reflectance model of the method, the fit of the field to its
training views, the fields accepted by the training-view check (tau = 40 dB) and the resulting system output (fused output if
accepted, single-image prediction otherwise), and the physical branch with the extended reflectance model
(--brdf-ext both of scripts/run_sensitivity_unfrozen.py) where it was run.  Writes outputs/v2/material_mismatch.json.

    python scripts/summarize_material_mismatch.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "outputs" / "v2" / "sensitivity_unfrozen"
DATA = ROOT / "data" / "world_benchmark_materials"
VARIANTS = ("control", "aniso", "clearcoat", "sheen", "wrap", "oren")
TAU = 40.0


def run(n: int, variant: str, ext: bool) -> dict | None:
    stem = ("" if n == 70 else f"s00{n}_") + f"none_none_seed0_mat_{variant}" + ("_ext" if ext else "")
    p = U / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    fit = d["fit_train"]["full_psnr"]
    out = {"phys": d["phys"]["psnr"], "fuse": d["fuse"]["psnr"], "single": d["single"]["psnr"], "fit": fit, "accepted": fit >= TAU}
    out["system"] = out["fuse"] if out["accepted"] else out["single"]
    return out


def images(variant: str, n: int, key: str) -> np.ndarray:
    d = DATA / variant / "scenes" / "test" / f"scene_00{n}"
    frames = json.loads((d / "transforms.json").read_text(encoding="utf-8"))["frames"]
    return np.stack([np.asarray(Image.open(d / f[key]).convert("RGB")).astype(np.float64) / 255 for f in frames])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([10 * np.log10(1 / max(float(np.mean((x - y) ** 2)), 1e-12)) for x, y in zip(a, b)]))


def main() -> None:
    scenes = list(range(70, 80))
    summary = {"tau": TAU, "variants": {}}
    for v in VARIANTS:
        base = {n: run(n, v, False) for n in scenes}
        ext = {n: run(n, v, True) for n in scenes}
        base = {n: r for n, r in base.items() if r}
        ext = {n: r for n, r in ext.items() if r}
        entry = {"n": len(base), "per_scene": {str(n): {"base": base[n], "extended": ext.get(n)} for n in base}}
        if base:
            arr = lambda key, rows: np.array([r[key] for r in rows])
            rows = list(base.values())
            entry.update({k: float(arr(k, rows).mean()) for k in ("phys", "fuse", "single", "system")})
            entry.update(phys_min=float(arr("phys", rows).min()), phys_max=float(arr("phys", rows).max()),
                         fit_min=float(arr("fit", rows).min()), fit_max=float(arr("fit", rows).max()),
                         accepted=int(sum(r["accepted"] for r in rows)))
        if ext:
            common = sorted(set(ext) & set(base))
            entry["extended"] = {"n": len(common), "scenes": common,
                                 "phys": float(np.mean([ext[n]["phys"] for n in common])),
                                 "phys_min": float(min(ext[n]["phys"] for n in common)), "phys_max": float(max(ext[n]["phys"] for n in common)),
                                 "fuse": float(np.mean([ext[n]["fuse"] for n in common])),
                                 "fit_min": float(min(ext[n]["fit"] for n in common)), "fit_max": float(max(ext[n]["fit"] for n in common)),
                                 "base_phys_same_scenes": float(np.mean([base[n]["phys"] for n in common])),
                                 "accepted": int(sum(ext[n]["accepted"] for n in common))}
        if v != "control":
            entry["input_vs_control"] = float(np.mean([psnr(images(v, n, "file_path"), images("control", n, "file_path")) for n in scenes]))
            entry["diffuse_vs_control"] = float(np.mean([psnr(images(v, n, "diffuse_path"), images("control", n, "diffuse_path")) for n in scenes]))
        summary["variants"][v] = entry
        line = f"{v:10s} n={entry['n']:2d}"
        if "phys" in entry:
            line += f" phys {entry['phys']:6.2f} ({entry['phys_min']:.2f}-{entry['phys_max']:.2f}) fit {entry['fit_min']:.1f}-{entry['fit_max']:.1f} accepted {entry['accepted']} system {entry['system']:.2f}"
        if "extended" in entry:
            e = entry["extended"]
            line += f" | extended ({e['n']} scenes) {e['phys']:.2f} vs {e['base_phys_same_scenes']:.2f}"
        if "input_vs_control" in entry:
            line += f" | input vs control {entry['input_vs_control']:.2f} dB"
        print(line)
    (ROOT / "outputs" / "v2" / "material_mismatch.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
