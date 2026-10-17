#!/bin/bash
# Build a Python 3.6 / PyTorch 1.5.1 (CPU) environment for the JSHDR release (CPython 3.6, Linux x86_64 only)
set -e
cd /root/jshdr
if [ ! -d /root/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda3 > /dev/null
fi
source /root/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^jshdr '; then
  conda create -y -q -n jshdr -c conda-forge --override-channels python=3.6 pip > /dev/null
fi
conda activate jshdr
python --version
pip install -q --no-cache-dir numpy==1.19.5 pillow==8.4.0 > /dev/null
pip install -q --no-cache-dir torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html > /dev/null
python -c "import torch, torchvision, numpy, PIL; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'numpy', numpy.__version__, 'PIL', PIL.__version__)"
# the release of JSHDR (compiled modules + inference script) and the checkpoint
if [ ! -d /root/jshdr/SHIQ ]; then
  git clone -q https://github.com/KosRud/SHIQ.git /root/jshdr/SHIQ
fi
mkdir -p /root/jshdr/SHIQ/network/checkpoints
if [ ! -d /root/jshdr/SHIQ/network/checkpoints/jshdr ]; then
  unzip -q -o /root/jshdr/jshdr.zip -d /root/jshdr/SHIQ/network/checkpoints/
fi
ls -la /root/jshdr/SHIQ/network/checkpoints/ | head
ls -la /root/jshdr/SHIQ/network/utils/
echo SETUP_DONE
