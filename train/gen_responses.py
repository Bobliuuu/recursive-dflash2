"""Regenerate responses with the target (Qwen3-8B, thinking OFF) through a running SGLang server (AR). Resumable (appends).
Usage: python gen_responses.py prompts.jsonl out.jsonl [--conc 256 --max-new 1536 --temp 0.7]
"""
import argparse, json, os, asyncio, aiohttp
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:30000")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--conc", type=int, default=256)
ap.add_argument("--max-new", type=int, default=1536)
ap.add_argument("--temp", type=float, default=0.7)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(a.prompts)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
rows = [r for r in rows if r["id"] not in done]
print("todo", len(rows), "done", len(done), flush=True)
fout = open(a.out, "a")
cnt = 0
async def one(sess, sem, r):
    global cnt
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    sp = {"temperature": a.temp, "top_p": 0.8, "top_k": 20, "max_new_tokens": a.max_new}
    async with sem:
        try:
            async with sess.post(a.url + "/generate", json={"text": text, "sampling_params": sp}, timeout=aiohttp.ClientTimeout(total=7200)) as resp:
                o = await resp.json()
        except Exception as e:
            print("err", repr(e), flush=True); return
    mi = o["meta_info"]
    fout.write(json.dumps({"id": r["id"], "source": r["source"], "prompt": r["prompt"], "response": o["text"],
                           "n_out": mi["completion_tokens"], "truncated": mi.get("finish_reason", {}).get("type") == "length"}) + "\n")
    cnt += 1
    if cnt % 500 == 0: fout.flush(); print("gen", cnt, flush=True)
async def main():
    sem = asyncio.Semaphore(a.conc)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as sess:
        await asyncio.gather(*[one(sess, sem, r) for r in rows])
asyncio.run(main())
fout.close()
print("finished", cnt)
