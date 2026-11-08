# Phase 1 on Qwen3.8-27B: official drafters, one box, one engine

Date: 2026-11-08. Hardware: 1x H100-80GB (Vast), bf16 target. Engine: SGLang 0.5.22 (torch 2.14.1+cu130, FlashInfer 0.7.1), Tencent `sglang-dspine` 0.2.0 plugin
(commit 6450af7) for DSpine. Everything below is raw-JSON backed: `bench/results/qwen38_27b/`, logs in `logs/qwen38_27b/`,
scripts in `bench/qwen38_27b/` (`serve.sh`, `client.py`, `sweep*.sh`, `queue*.sh`, `summarize.py`).

RESULTS_PLACEHOLDER
