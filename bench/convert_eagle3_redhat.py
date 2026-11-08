"""Convert RedHatAI/Qwen3-8B-speculator.eagle3 (speculators format) into the SGLang EAGLE3 layout
(LlamaForCausalLMEagle3: layers.0.* -> midlayer.*, embed_tokens dropped; config from AngelSlim's Qwen3-8B_eagle3, same shapes).
Usage: python convert_eagle3_redhat.py <out_dir>"""
import sys, glob, json, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
HF = os.environ.get("HF_HOME", "/workspace/hf") + "/hub"
src = glob.glob(f"{HF}/models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/*/model.safetensors")[0]
cfg = glob.glob(f"{HF}/models--AngelSlim--Qwen3-8B_eagle3/snapshots/*/config.json")[0]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
sd = {}
with safe_open(src, "pt") as g:
    for k in g.keys():
        if k == "embed_tokens.weight": continue
        sd[k.replace("layers.0.", "midlayer.")] = g.get_tensor(k).contiguous()
save_file(sd, out + "/model.safetensors")
shutil.copy(cfg, out + "/config.json")
print("wrote", out, sorted(sd))
