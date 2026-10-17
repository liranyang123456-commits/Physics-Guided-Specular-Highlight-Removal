"""Matched domain fine-tuning on SSHR, NSH, and PSD: every method scored with the same code and targets on the same images.

Rows per dataset: the raw input; our branch zero-shot (SHIQ checkpoint outputs/shiq_mask) and fine-tuned
(scripts/train_paired_pro.py); TSHRNet and DHAN-SHR zero-shot (released checkpoints) and fine-tuned with the same protocol
(scripts/finetune_baselines_external.py).  Test images: SSHR, the first 2000 of the test split (tone-corrected targets);
NSH, the 360 test images; PSD, the 420 images of the held-out groups (ft_test) of the 2060 pairs of the archive, on which our
branch is fine-tuned again (outputs/psd_finetuned_current) because the first submission evaluated only 1803 of these pairs.
The scores of our branch come from scripts/eval_paired_dataset.py (outputs/rescore_external/).  The zero-shot SSHR scores of the
baselines are also checked against the first-submission runs (outputs/baselines/*_sshr).
Writes outputs/v2/external_ft_summary.json.

    python scripts/summarize_external_ft.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RS = OUT / "rescore_external"
SETS = {"sshr": ("test_first2000", "ours_zs_sshr_first2000", "ours_ft_sshr_first2000"),
        "nsh": ("test", "ours_zs_nsh_test", "ours_ft_nsh_test"),
        "psd": ("ft_test", "ours_zs_psd_ft_test", "ours_ft_psd_ft_test")}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    rows = {}
    for ds, (split, zs_dir, ft_dir) in SETS.items():
        zs, ft = load(RS / zs_dir / "metrics.json"), load(RS / ft_dir / "metrics.json")
        assert zs["n"] == ft["n"] and abs(zs["input_psnr"] - ft["input_psnr"]) < 1e-9
        r = {"n": ft["n"], "input": {"psnr": ft["input_psnr"], "ssim": ft["input_ssim"]},
             "ours_zs": {"psnr": zs["ours_psnr"], "ssim": zs["ours_ssim"]}, "ours_ft": {"psnr": ft["ours_psnr"], "ssim": ft["ours_ssim"]}}
        for m in ("tshrnet", "dhan_shr"):
            d = load(OUT / "baselines_external_ft" / f"{m}_{ds}" / "metrics.json")
            assert d["zero_shot"][split]["n"] == ft["n"] == d["test"][split]["n"], (ds, m, d["zero_shot"][split]["n"], ft["n"])
            r[f"{m}_zs"], r[f"{m}_ft"] = d["zero_shot"][split], d["test"][split]
            r[f"{m}_crop"], r[f"{m}_best_step"] = d.get("train_crop"), d["validation_best"]["step"]
        rows[ds] = r
        print(ds, {k: (round(v["psnr"], 2) if isinstance(v, dict) else v) for k, v in r.items()})
    full = {"ours_zs": load(RS / "ours_zs_psd_test" / "metrics.json"), "dhan_zs": load(RS / "dhan_zs_psd_test" / "metrics.json"),
            "tshrnet_zs": load(RS / "tshrnet_zs_psd_test" / "metrics.json")}
    rows["psd_full_test"] = {"n": full["ours_zs"]["n"], "input": {"psnr": full["ours_zs"]["input_psnr"], "ssim": full["ours_zs"]["input_ssim"]},
                             "ours_zs": {"psnr": full["ours_zs"]["ours_psnr"], "ssim": full["ours_zs"]["ours_ssim"]},
                             "dhan_zs": {"psnr": full["dhan_zs"]["psnr"], "ssim": full["dhan_zs"]["ssim"]},
                             "tshrnet_zs": {"psnr": full["tshrnet_zs"]["psnr"], "ssim": full["tshrnet_zs"]["ssim"]}}
    print("psd full test", {k: (round(v["psnr"], 2) if isinstance(v, dict) else v) for k, v in rows["psd_full_test"].items()})
    check = {}
    for m in ("tshrnet", "dhan_shr"):
        first = load(OUT / "baselines" / f"{m}_sshr" / "metrics.json")
        rerun = load(RS / f"{m.split('_')[0]}_zs_sshr_first2000" / "metrics.json")
        check[m] = {"first_submission": first["psnr"], "rerun": rerun["psnr"], "finetune_script": rows["sshr"][f"{m}_zs"]["psnr"]}
        assert first["n"] == rerun["n"] == 2000 and max(check[m].values()) - min(check[m].values()) < 0.01, check[m]
    rows["sshr_zero_shot_check"] = check
    print("sshr zero-shot check", check)
    (ROOT / "outputs/v2/external_ft_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
