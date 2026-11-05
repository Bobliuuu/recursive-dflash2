"""Roofline: time of one target forward over B new tokens with a KV cache of length CTX.
Usage: python verify_sweep.py MODEL_DIR [CTX]"""
import sys, time, json, torch
from transformers import AutoModelForCausalLM
path = sys.argv[1]; CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 512
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
dev = "cuda"; out = {}
with torch.inference_mode():
    prompt = torch.randint(1000, 20000, (1, CTX), device=dev)
    for B in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        ts = []
        for it in range(25):
            o = m(prompt, use_cache=True); pkv = o.past_key_values
            x = torch.randint(1000, 20000, (1, B), device=dev)
            torch.cuda.synchronize(); t = time.perf_counter()
            m(x, past_key_values=pkv, use_cache=True)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            del pkv, o
        ts = sorted(ts[5:]); out[B] = ts[len(ts) // 2] * 1e3
        print(f"B={B:4d}  {out[B]:7.2f} ms  ({out[B]/out[1]:.2f}x of B=1)", flush=True)
json.dump(out, open(f"verify_sweep_ctx{CTX}.json", "w"))
