#!/bin/bash
# Fresh-box setup for the Qwen3-8B / DSpine benchmarks (1x A100, Vast image nvidia/cuda:12.4.1-devel-ubuntu22.04).
# Everything lives under /workspace. Writes /workspace/env.sh (sourced by all bench/*.sh scripts).
# Usage (on the box, repo cloned at /workspace/recursive-dflash2): bash bench/setup_box.sh
set -euo pipefail
W=/workspace
mkdir -p $W/hf $W/p2 $W/p4
cd $W
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
apt-get -qq update >/dev/null 2>&1 || true; apt-get -qq install -y git tmux curl >/dev/null 2>&1 || true
[ -d $W/venv ] || uv venv -p 3.12 $W/venv
source $W/venv/bin/activate
[ -d $W/DSpine ] || git clone -q https://github.com/Tencent/DSpine $W/DSpine
# DSpine pulls sglang 0.5.22 / torch 2.14.1+cu130 / flashinfer 0.7.1
uv pip install -q -e $W/DSpine
# flashinfer / sgl-kernel JIT need a CUDA 13.0 toolchain matching the 13.0 runtime headers (box nvcc 12.4 is too old; 13.5 nvcc vs 13.0 headers fails)
uv pip install -q "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" ninja transformers huggingface_hub
SP=$(python -c "import site; print(site.getsitepackages()[0])")
CU=$SP/nvidia/cu13
[ -e $CU/lib64 ] || ln -s lib $CU/lib64
[ -e $CU/lib/libcudart.so ] || ln -s libcudart.so.13 $CU/lib/libcudart.so
cat > $W/env.sh <<EOF
source $W/venv/bin/activate
export HF_HOME=$W/hf
export CUDA_HOME=$CU
export PATH=$CU/bin:\$PATH
export LD_LIBRARY_PATH=$CU/lib:\${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.0
export SGLANG_DISABLE_REQUEST_LOGGING=1
EOF
source $W/env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["Qwen/Qwen3-8B", "tencent/Qwen3-8B-DSpine-b16"]:
    print(snapshot_download(r))
PY
# draft dirs with block_size N (weights untouched; the b16 checkpoint run at a longer block / tree width)
DSP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('tencent/Qwen3-8B-DSpine-b16'))")
for N in 16 32 48; do
  D=$W/dspine_b$N; mkdir -p $D
  for f in $DSP/*; do [ "$(basename $f)" = config.json ] || ln -sf $f $D/; done
  python -c "import json; c=json.load(open('$DSP/config.json')); c['block_size']=$N; json.dump(c, open('$D/config.json','w'), indent=1)"
done
python $W/recursive-dflash2/bench/patches/apply_timing_patch.py
python -c "import torch, sglang, flashinfer; print(torch.__version__, sglang.__version__, flashinfer.__version__)"
echo setup ok
