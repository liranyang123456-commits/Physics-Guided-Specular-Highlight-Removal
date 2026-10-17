"""Run the released JSHDR inference script (compiled for CPython 3.6) on a CPU-only PyTorch 1.5.1.

The release loads its CUDA-saved checkpoint with torch.load(...) without map_location and may call .cuda(); the shim maps
storages to the CPU and turns .cuda() into a no-op.  Nothing else is changed (the compiled modules do the computation).
usage: python jshdr_cpu_shim.py -c jshdr -i image.png -r out_dir
"""
import runpy
import sys

import torch

_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _orig_load(*args, **kwargs)


torch.load = _load
torch.nn.Module.cuda = lambda self, device=None: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self
sys.argv = ["infer.py"] + sys.argv[1:]
runpy.run_path("infer.py", run_name="__main__")
