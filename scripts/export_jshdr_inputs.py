"""Write the inputs on which the released JSHDR checkpoint is evaluated (it runs in WSL, see scripts/jshdr_batch.sh).

The inputs are exactly those that the other baselines see: the 1,000 official SHIQ test images at 200 x 200 (the same
bilinear loading as SHIQAdapter) and the 240 test frames of world_benchmark_v2 at 128 x 128.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHIQ_ROOT = ROOT / "data" / "raw" / "SHIQ_extracted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)

    # SHIQ: ids 14001-15000 of the official test directory (SHIQAdapter split "test_full")
    from mvbrdf_shr.data.shiq_adapter import SHIQAdapter  # imports the package; run with the full repo

    dataset = SHIQAdapter(SHIQ_ROOT, 200, "test_full", augment=False)
    target = out / "shiq_in"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for a, d, _, _ in dataset.pairs:
        Image.open(a).convert("RGB").resize((200, 200), Image.BILINEAR).save(target / a.name)
    print("shiq inputs:", len(dataset.pairs))

    # world_benchmark_v2 test frames
    bench = ROOT / "data" / "world_benchmark_v2" / "flat" / "test" / "input"
    target = out / "bench_in"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    n = 0
    for path in sorted(bench.glob("*.png")):
        shutil.copy2(path, target / path.name)
        n += 1
    print("benchmark inputs:", n)


if __name__ == "__main__":
    main()
