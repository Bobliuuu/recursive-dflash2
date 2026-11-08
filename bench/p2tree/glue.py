"""Glue between SGLang's DFlash/DSpine v2 worker and the tree kernels (bs=1, greedy). Env: RDF2_TREE_SPEC="N:D" (N must equal the verify block size). Installed by apply_tree_patch.py."""
import os, torch

class TreeGlue:
    def __init__(self):
        from tree_kernels import TreeVerifier
        n, d = os.environ["RDF2_TREE_SPEC"].split(":")   # "N:D" = verify width N (= block size), max tree depth D
        self.tv = TreeVerifier(int(n), int(d))
        self.N = self.tv.N; self.D = self.tv.D
        self.kv_ready = False
        self.stats = [0, 0]
        self.commit_pos_buf = torch.zeros(self.N, dtype=torch.int64, device="cuda")
        self.ar = torch.arange(self.N, dtype=torch.int64, device="cuda")
        self.dump = os.environ.get("RDF2_TREE_STATS")

    # ---- table from the DSpine sampler (static buffers after the draft graph replay)
    def table(self, sampler):
        from sglang_dspine.kernels import dspine as dk
        bridge = sampler.draft_model.selected_prefix_bridge
        (t, k) = (sampler.block_size, sampler.k)
        m, n = t, t - 1
        cand = sampler.candidate_out[:1].view(n, k)
        tab = dk.serial_rs_table(sampler.raw[:m], sampler.rms[:m], sampler.query[:m], sampler.anchors[:1], cand, bridge.codebook,
                                 sampler.pred_gate_f32, sampler.prev_f32, sampler.gate_b_f32, sampler.write_f32, sampler.norm_f32,
                                 sampler.lm_head_weight, sampler.norm_eps, t, res_emb=sampler.res_emb, res_proj=sampler.res_proj_f32)
        return tab, cand

    def prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d):
        assert bs == 1 and int(worker.block_size) == self.N, (bs, worker.block_size, self.N)
        tab, cand = self.table(worker._draft_sampler)
        pl = prefix_lens[:1].to(torch.int64)
        self.tv.build(tab[: self.D], cand[: self.D], block_ids[0, 0:1].to(torch.int64), pl)
        draft_tokens[0].copy_(self.tv.tokens)
        positions_2d[0].copy_(self.tv.positions)
        torch.add(pl, self.ar, out=self.commit_pos_buf)
        return self

    def mask(self):
        return self.tv.fmask.view(torch.bool)

    def accept(self, worker, logits_output, kv_loc_plan, prefix_lens):
        tv = self.tv
        if not self.kv_ready:
            pool = worker.target_worker.model_runner.token_to_kv_pool
            tv.set_kv(pool.k_buffer, pool.v_buffer); self.kv_ready = True
        am = torch.argmax(logits_output.next_token_logits, dim=-1)
        loc = kv_loc_plan.write_ids(worker.target_worker.model_runner.kv_index_translator).reshape(-1).to(torch.int64)
        tv.accept(am, loc, prefix_lens[:1].to(torch.int64))
        hs = logits_output.hidden_states
        logits_output.hidden_states = hs.view(self.N, -1).index_select(0, tv.path)
        if self.dump:
            self.stats[0] += 1; self.stats[1] += int(tv.accept_len.item()) + 1
            if self.stats[0] % 200 == 0:
                open(self.dump, "a").write(f"{self.stats[0]} rounds mean_emit {self.stats[1]/self.stats[0]:.3f}\n")
        return tv.accept_len, tv.commit_lens, tv.bonus, tv.out_tokens.view(1, self.N), tv.new_seq_lens

def chain_mask(N, prefix_lens_host, device="cuda"):
    """causal verify mask for the non-tree fallback: per request [N rows x (P+N)] allowed matrix, flattened."""
    parts = []
    for P in prefix_lens_host:
        P = int(P)
        parts.append(torch.cat([torch.ones(N, P, dtype=torch.bool, device=device), torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))], 1).reshape(-1))
    return torch.cat(parts)
