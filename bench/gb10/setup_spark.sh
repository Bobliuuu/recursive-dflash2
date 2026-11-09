#!/bin/bash
# DGX Spark (GB10, aarch64, sm_121, CUDA 13.0 driver) setup without root: everything under $R (default ~/rdf2), uv-managed Python
# (the system python3.12 has no headers, which breaks Triton's driver build), pip wheels for torch 2.14.1+cu130 / sglang 0.5.22 /
# flashinfer 0.7.1 (all have aarch64 builds; flashinfer JIT-compiles its sm_121a kernels on first use, ~3 min).
# Usage: bash setup_spark.sh   (repo cloned at $R/recursive-dflash2)
set -euo pipefail
R=${R:-$HOME/rdf2}; mkdir -p $R/.cache $R/tmp $R/logs; cd $R
UV=${UV:-$HOME/.local/bin/uv}
cat > $R/env.sh <<EOS
export HF_HOME=$R/hf XDG_CACHE_HOME=$R/.cache UV_CACHE_DIR=$R/.cache/uv PIP_CACHE_DIR=$R/.cache/pip TRITON_CACHE_DIR=$R/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor CUDA_CACHE_PATH=$R/.cache/nv FLASHINFER_WORKSPACE_BASE=$R/.cache TORCH_EXTENSIONS_DIR=$R/.cache/torch_extensions
export TMPDIR=$R/tmp UV_PYTHON_INSTALL_DIR=$R/.uvpython SGLANG_DISABLE_REQUEST_LOGGING=1 TORCH_CUDA_ARCH_LIST=12.1
EOS
source $R/env.sh
[ -d $R/venv ] || $UV venv -q -p 3.12 --managed-python $R/venv
source $R/venv/bin/activate
[ -d $R/DSpine ] || git clone -q https://github.com/Tencent/DSpine $R/DSpine
$UV pip install -q -e $R/DSpine "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" ninja transformers huggingface_hub
SP=$(python -c "import site; print(site.getsitepackages()[0])"); CU=$SP/nvidia/cu13
[ -e $CU/lib64 ] || ln -s lib $CU/lib64
[ -e $CU/lib/libcudart.so ] || ln -s libcudart.so.13 $CU/lib/libcudart.so
cat >> $R/env.sh <<EOS
source $R/venv/bin/activate
export CUDA_HOME=$CU PATH=$CU/bin:\$PATH LD_LIBRARY_PATH=$CU/lib:\${LD_LIBRARY_PATH:-}
EOS
source $R/env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["Qwen/Qwen3-8B", "tencent/Qwen3-8B-DSpine-b16", "z-lab/Qwen3-8B-DFlash-b16"]:
    print(snapshot_download(r))
PY
DSP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('tencent/Qwen3-8B-DSpine-b16'))")
for N in 16 24 32 48 64 96; do
  D=$R/dspine_b$N; mkdir -p $D
  for f in $DSP/*; do [ "$(basename $f)" = config.json ] || ln -sf $f $D/; done
  python -c "import json; c=json.load(open('$DSP/config.json')); c['block_size']=$N; json.dump(c, open('$D/config.json','w'), indent=1)"
done
python $R/recursive-dflash2/bench/patches/apply_timing_patch.py
python -c "import torch, sglang, flashinfer; print(torch.__version__, sglang.__version__, flashinfer.__version__, torch.cuda.get_device_capability())"
echo "setup ok; add RDF2_PORT=<free local port> to $R/env.sh"
