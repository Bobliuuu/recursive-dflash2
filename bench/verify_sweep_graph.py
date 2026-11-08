"""Roofline with CUDA graphs: one target forward over B new tokens, static KV cache of CTX.
Usage: python verify_sweep_graph.py MODEL_DIR [CTX]"""
import sys, time, json, torch
from transformers import AutoModelForCausalLM, StaticCache
path = sys.argv[1]; CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 512
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
dev = "cuda"; out = {}
Bs = [int(a) for a in sys.argv[3:]] or [1]
MAXB = 512
with torch.inference_mode():
    prompt = torch.randint(1000, 20000, (1, CTX), device=dev)
    cache = StaticCache(config=m.config, max_cache_len=CTX + max(Bs) + 8)
    m(prompt, past_key_values=cache, use_cache=True, cache_position=torch.arange(CTX, device=dev))
    cls = [l.cumulative_length for l in cache.layers]
    def reset():
        for c in cls: c.fill_(CTX)
    torch.cuda.synchronize()
    for B in Bs:
        x = torch.randint(1000, 20000, (1, B), device=dev)
        pos = torch.arange(CTX, CTX + B, device=dev)
        L = CTX + max(Bs) + 8
        mask = (torch.arange(L, device=dev)[None, :] <= pos[:, None])[None, None]  # explicit 4D mask: no host-sync mask creation
        for _ in range(3):
            reset(); m(x, past_key_values=cache, cache_position=pos, use_cache=True, logits_to_keep=B, attention_mask=mask)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            reset()
            y = m(x, past_key_values=cache, cache_position=pos, use_cache=True, logits_to_keep=B, attention_mask=mask)
        for _ in range(5): g.replay()
        ts = []
        for _ in range(30):
            torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts.sort(); out[B] = ts[len(ts) // 2] * 1e3
        print(f"B={B:4d}  {out[B]:7.2f} ms  ", flush=True)
        torch.cuda.synchronize(); del g
json.dump(out, open(f"vg_ctx{CTX}_B{Bs[0]}.json", "w"))
