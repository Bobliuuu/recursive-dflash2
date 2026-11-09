"""Output identity between two runs (same prompts): fraction of requests with byte-identical text and identical token counts.
Usage: python p4_compare_text.py PREFIX_A PREFIX_B   (bench/results/PREFIX_<ds>_T0_c1.json)"""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
a, b = sys.argv[1:3]
tot = same = 0
for ds in ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]:
    ra = json.load(open(f"{R}/{a}_{ds}_T0_c1.json"))["requests"]; rb = json.load(open(f"{R}/{b}_{ds}_T0_c1.json"))["requests"]
    s = sum(x["text_sha"] == y["text_sha"] for x, y in zip(ra, rb))
    print(f"{ds:10s} identical {s}/{len(ra)}")
    tot += len(ra); same += s
print(f"all        identical {same}/{tot}")
