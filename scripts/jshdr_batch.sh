#!/bin/bash
# Run the released JSHDR checkpoint (CPython 3.6 / PyTorch 1.5.1, CPU) on every PNG of a folder.
# usage: jshdr_batch.sh INPUT_DIR OUTPUT_DIR [PARALLEL_JOBS]
# The release (github.com/KosRud/SHIQ, network/infer.py) loads the model once per image; the shim only maps the CUDA-saved
# checkpoint to the CPU (see jshdr_cpu_shim.py).  Outputs are D_<input name>.png.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate jshdr
IN=$1
OUT=$2
JOBS=${3:-6}
mkdir -p "$OUT"
cd /root/jshdr/SHIQ/network
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ls "$IN"/*.png | xargs -P "$JOBS" -I{} python jshdr_cpu_shim.py -c jshdr -i {} -r "$OUT" > /dev/null 2>&1
echo "outputs: $(ls "$OUT" | wc -l)"
