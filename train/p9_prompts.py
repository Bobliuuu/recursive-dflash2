"""Phase 9 prompt sets (box). Two steps:

  python p9_prompts.py domains OUTDIR --frozen bench/data/prompts
      Per-domain proxy-eval prompts, formatted exactly like bench/prep_prompts.py, DISJOINT from the frozen benchmark
      files (all 100 ids per dataset are excluded, not only the 40 that run_cell.py uses):
        eval_{gsm8k,math500,humaneval,mbpp}.prompts.jsonl  (64 each, seeded)
      plus code training prompts from the MBPP train/validation/prompt splits (disjoint from the MBPP test set):
        mbpp_train.prompts.jsonl
      MT-Bench has only 80 first-turn prompts and all 80 are frozen, so its proxy stand-in is "chat" (see `split`).

  python p9_prompts.py split GEN.jsonl MBPP_TRAIN_GEN.jsonl OUTDIR
      Splits the generated open-perfectblend responses (train/prep_prompts.py prompts) into
        train_general.jsonl (all but the last 256), eval_general.jsonl (last 256, never trained on),
        eval_chat.jsonl (eval_general rows whose source is neither math nor code, up to 64),
        train_code.jsonl (train_general rows with a code source + MBPP train responses).
"""
import json, os, random, sys, collections
MATHSUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def is_code(src): return "code" in src.lower()
def is_math(src): s = src.lower(); return "math" in s or "gsm" in s or "orca" in s


def domains(out, frozen, n=64):
    from datasets import load_dataset
    os.makedirs(out, exist_ok=True)
    sets = {}
    sets["gsm8k"] = [r["question"] + MATHSUF for r in load_dataset("openai/gsm8k", "main", split="test")]
    sets["math500"] = [r["problem"] + MATHSUF for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    sets["humaneval"] = ["Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
                         + r["prompt"] + "\n```" for r in load_dataset("openai/openai_humaneval", split="test")]
    mb = lambda r: r["text"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"])
    sets["mbpp"] = [mb(r) for r in load_dataset("google-research-datasets/mbpp", "full", split="test")]
    for k, v in sets.items():
        fz = {json.loads(l)["id"] for l in open(f"{frozen}/{k}.jsonl")}
        fz_text = {json.loads(l)["prompt"] for l in open(f"{frozen}/{k}.jsonl")}
        idx = [i for i in range(len(v)) if i not in fz and v[i] not in fz_text]
        random.Random(9).shuffle(idx)
        idx = sorted(idx[:n])
        with open(f"{out}/eval_{k}.prompts.jsonl", "w") as f:
            for i in idx: f.write(json.dumps({"id": i, "source": k, "prompt": v[i]}) + "\n")
        print(k, "total", len(v), "frozen", len(fz), "eval", len(idx))
    rows = []
    for split in ("train", "validation", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            rows.append({"id": 10_000_000 + int(r["task_id"]), "source": "mbpp_" + split, "prompt": mb(r)})
    with open(f"{out}/mbpp_train.prompts.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print("mbpp train/validation/prompt", len(rows))


def split(gen, mbpp_gen, out, hold=256, nchat=64):
    rows = [json.loads(l) for l in open(gen)]
    tr, ev = rows[:-hold], rows[-hold:]
    def dump(name, rs):
        with open(f"{out}/{name}", "w") as f:
            for r in rs: f.write(json.dumps(r) + "\n")
        print(name, len(rs))
    dump("train_general.jsonl", tr)
    dump("eval_general.jsonl", ev)
    dump("eval_chat.jsonl", [r for r in ev if not is_code(r["source"]) and not is_math(r["source"])][:nchat])
    code = [r for r in tr if is_code(r["source"])] + [json.loads(l) for l in open(mbpp_gen)]
    random.Random(5).shuffle(code)
    dump("train_code.jsonl", code)
    print("train sources", collections.Counter(r["source"] for r in tr).most_common(20))
    print("code sources", collections.Counter(r["source"] for r in code).most_common(20))


if __name__ == "__main__":
    if sys.argv[1] == "domains":
        domains(sys.argv[2], sys.argv[sys.argv.index("--frozen") + 1] if "--frozen" in sys.argv else "bench/data/prompts")
    else:
        split(*sys.argv[2:5])
