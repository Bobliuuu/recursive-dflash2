"""Phase 9 on Modal: one A100-80GB container per stage, same software stack as bench/setup_box.sh
(DSpine plugin -> SGLang 0.5.22 / torch 2.14.1+cu130 / flashinfer 0.7.1, CUDA 13.0 JIT toolchain) plus SpecForge b15ad38.
State (HF cache, data, masks, results, logs) lives on the Modal Volume `rdf2-p9deletion` under /vol.

  modal run --detach bench/p9deletion/modal_app.py --stage smoke|data|train|eval|serve
  modal volume get rdf2-p9deletion p9/eval/proxy.json bench/results/p9deletion/
"""
import os
import subprocess

import modal

DSPINE_COMMIT = "321b0fc1388c4cdf03483ab73856975f7c0c0be7"
SPECFORGE_COMMIT = "b15ad38"
R = "/workspace/recursive-dflash2"
SP = "/usr/local/lib/python3.12/site-packages"
CU = f"{SP}/nvidia/cu13"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "procps")
    .pip_install("uv")
    .run_commands(
        f"git clone -q https://github.com/Tencent/DSpine /opt/DSpine && cd /opt/DSpine && git checkout -q {DSPINE_COMMIT}",
        "uv pip install --system -e /opt/DSpine",
        'uv pip install --system "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" ninja transformers huggingface_hub',
        f"[ -e {CU}/lib64 ] || ln -s lib {CU}/lib64",
        f"[ -e {CU}/lib/libcudart.so ] || ln -s libcudart.so.13 {CU}/lib/libcudart.so",
        f"git clone -q https://github.com/sgl-project/SpecForge /opt/SpecForge && cd /opt/SpecForge && git checkout -q {SPECFORGE_COMMIT}",
        "uv pip install --system --no-deps -e /opt/SpecForge",
        'uv pip install --system --no-deps "liger-kernel>=0.8.3,<0.9" yunchang',
        "uv pip install --system datasets accelerate click openai-harmony pydantic psutil pyyaml tensorboard aiohttp pandas pytest",
    )
    .env({
        "CUDA_HOME": CU,
        "PATH": f"{CU}/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": f"{CU}/lib",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "SGLANG_DISABLE_REQUEST_LOGGING": "1",
        "HF_HOME": "/vol/hf",
        "R": R,
        "P9": "/vol/p9",
    })
    .add_local_dir(f"{REPO}/train", f"{R}/train", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(f"{REPO}/bench", f"{R}/bench", ignore=["results", "results/**", "__pycache__", "*.pyc", "gb10", "qwen38_27b"])
)

app = modal.App("rdf2-p9deletion", image=image)
vol = modal.Volume.from_name("rdf2-p9deletion", create_if_missing=True)

SMOKE = r"""
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch, sglang, flashinfer, specforge, sglang_dspine, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), sglang.__version__, flashinfer.__version__, transformers.__version__)"
python -c "import torch; x=torch.randn(256,256,device='cuda'); print('matmul ok', float((x@x).sum())!=0)"
python -c "from specforge.modeling.draft.dspine import DSpineDraftModel; from specforge.algorithms.dspine.model import OnlineDSpineModel; print('specforge dspine ok')"
cd $R && python -m pytest -q -p no:cacheprovider train/test_deletion_mask.py 2>&1 | tail -2
"""


@app.function(gpu="A100-80GB", timeout=5 * 3600, volumes={"/vol": vol}, cpu=6, memory=40960)
def run(stage: str, extra_env: dict | None = None) -> str:
    os.makedirs("/vol/p9/logs", exist_ok=True)
    env = dict(os.environ, **(extra_env or {}))
    cmd = ["bash", "-c", SMOKE] if stage == "smoke" else ["bash", f"{R}/bench/p9deletion/box_p9.sh", stage]
    log = f"/vol/p9/logs/stage_{stage}.log"
    try:
        with open(log, "a") as f:
            p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        out = open(log).read()[-6000:]
        return f"exit={p.returncode}\n{out}"
    finally:
        vol.commit()


@app.local_entrypoint()
def main(stage: str = "smoke", masks: str = "", steps: str = ""):
    extra = {}
    if masks: extra["MASKS"] = masks
    if steps: extra["STEPS"] = steps
    print(run.remote(stage, extra))
