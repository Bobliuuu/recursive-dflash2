"""CUDA kernels for in-engine tree verification (bs=1, greedy): beam-tree build from DSpine's [D,16,16] predecessor-conditioned logit table,
tree accept walk, paged-KV compaction of the accepted path. Built with torch load_inline (A100: TORCH_CUDA_ARCH_LIST=8.0)."""
import os, torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math.h>
#define MAXW 8
#define K 16


#define EXPAND 8
#define MAXN 128
__global__ void tree_build_kernel(const float* table, const long* cand, const long* anchor, const long* prefix_len,
                                  int D, int N, long* tokens, int* parent, int* depth, int* lane_out,
                                  unsigned char* tmask, unsigned char* fmask, long* positions) {
  // best-first tree: frontier of candidate children (score = joint prob), pop the best N-1 times.
  __shared__ float fscore[MAXN * EXPAND];
  __shared__ int fpar[MAXN * EXPAND];
  __shared__ int flane[MAXN * EXPAND];
  __shared__ int fdep[MAXN * EXPAND];
  __shared__ float nscore[MAXN];
  __shared__ int nlane[MAXN];
  __shared__ int ndep[MAXN];
  __shared__ int npar[MAXN];
  __shared__ float wval[4];
  __shared__ int widx[4];
  __shared__ int sel;
  __shared__ float rowp[16];
  __shared__ int rowrank[16];
  int t = threadIdx.x;
  int F = N * EXPAND;
  for (int i = t; i < F; i += blockDim.x) fscore[i] = -1.f;
  if (t == 0) { nscore[0] = 1.f; nlane[0] = 0; ndep[0] = 0; npar[0] = -1; tokens[0] = anchor[0]; }
  __syncthreads();
  int nn = 1;           // nodes so far
  int expand_node = 0;  // node to expand next (initially root)
  for (int it = 0; it < N; ++it) {
    // expand node `expand_node` -> push top-EXPAND lanes of its conditional row
    int d = ndep[expand_node];
    if (d < D) {
      if (t < 16) {
        int pl = (d == 0) ? 0 : nlane[expand_node];
        const float* row = table + ((size_t)d * 16 + pl) * 16;
        float mx = -1e30f;
        for (int k = 0; k < 16; ++k) mx = fmaxf(mx, row[k]);
        float den = 0.f;
        for (int k = 0; k < 16; ++k) den += __expf(row[k] - mx);
        float p = __expf(row[t] - mx) / den;
        rowp[t] = p;
      }
      __syncthreads();
      if (t < 16) {
        int rk = 0;
        for (int k = 0; k < 16; ++k) if (rowp[k] > rowp[t] || (rowp[k] == rowp[t] && k < t)) ++rk;
        if (rk < EXPAND) {
          int slot = expand_node * EXPAND + rk;
          fscore[slot] = nscore[expand_node] * rowp[t];
          fpar[slot] = expand_node; flane[slot] = t; fdep[slot] = d + 1;
        }
      }
      __syncthreads();
    }
    if (it == N - 1) break;
    // pop best
    float bv = -2.f; int bi = 0;
    for (int i = t; i < F; i += blockDim.x) { float v = fscore[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_down_sync(0xffffffff, bv, o); int oi = __shfl_down_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((t & 31) == 0) { wval[t >> 5] = bv; widx[t >> 5] = bi; }
    __syncthreads();
    if (t == 0) {
      float v = wval[0]; int ix = widx[0];
      for (int q = 1; q < 4; ++q) if (wval[q] > v || (wval[q] == v && widx[q] < ix)) { v = wval[q]; ix = widx[q]; }
      sel = ix;
      int node = nn;
      npar[node] = fpar[ix]; nlane[node] = flane[ix]; ndep[node] = fdep[ix]; nscore[node] = fscore[ix];
      tokens[node] = cand[(size_t)(fdep[ix] - 1) * 16 + flane[ix]];
      fscore[ix] = -1.f;
    }
    __syncthreads();
    expand_node = nn; nn++;
    __syncthreads();
  }
  for (int i = t; i < N; i += blockDim.x) { parent[i] = npar[i]; depth[i] = ndep[i]; lane_out[i] = nlane[i]; }
  for (int i = t; i < N * N; i += blockDim.x) tmask[i] = 0;
  __syncthreads();
  for (int i = t; i < N; i += blockDim.x) {
    int j = i;
    while (j >= 0) { tmask[i * N + j] = 1; j = npar[j]; }
  }
  __syncthreads();
  long P = prefix_len[0];
  long W = P + N;
  long nbits = (long)N * W;
  long nbytes = (nbits + 7) / 8;
  for (long j = t; j < nbytes; j += blockDim.x) {
    unsigned int byte = 0;
    for (int b = 0; b < 8; ++b) {
      long idx = j * 8 + b;
      unsigned char v = 0;
      if (idx < nbits) { long row = idx / W, col = idx % W; v = (col < P) ? 1 : tmask[row * N + (col - P)]; }
      byte |= ((unsigned int)v) << b;
    }
    fmask[j] = (unsigned char)byte;   // flashinfer packed custom mask (little bit order), request 0 at offset 0
  }
  for (int i = t; i < N; i += blockDim.x) positions[i] = P + ndep[i];
}

__global__ void tree_accept_kernel(const long* argmax, const long* tokens, const int* parent, int N,
                                   const long* loc, const long* prefix_len, long* out_tokens, int* accept_len, int* commit_lens,
                                   long* bonus, long* new_seq_lens, long* path, long* src_slot, long* dst_slot) {
  __shared__ int s_par[MAXN];
  __shared__ long s_tok[MAXN];
  __shared__ int s_cur, s_L, s_done;
  int t = threadIdx.x;
  for (int i = t; i < N; i += 32) { s_par[i] = parent[i]; s_tok[i] = tokens[i]; }
  if (t == 0) { s_cur = 0; s_L = 0; s_done = 0; path[0] = 0; }
  __syncthreads();
  while (!s_done) {
    long want = argmax[s_cur];
    int found = N;
    for (int i = t; i < N; i += 32) if (s_par[i] == s_cur && s_tok[i] == want) { found = min(found, i); }
    for (int o = 16; o > 0; o >>= 1) found = min(found, __shfl_down_sync(0xffffffff, found, o));
    found = __shfl_sync(0xffffffff, found, 0);
    if (t == 0) {
      if (found >= N) s_done = 1;
      else { s_cur = found; s_L++; path[s_L] = found; }
    }
    __syncthreads();
  }
  if (t == 0) {
    int L = s_L, cur = s_cur;
    long b = argmax[cur];
    for (int i = 0; i < N; ++i) out_tokens[i] = 0;
    for (int i = 0; i < L; ++i) out_tokens[i] = tokens[path[i + 1]];
    out_tokens[L] = b;
    for (int i = L + 1; i < N; ++i) path[i] = i;
    for (int i = 0; i < N; ++i) { src_slot[i] = loc[path[i]]; dst_slot[i] = loc[i]; }
    accept_len[0] = L; commit_lens[0] = L + 1; bonus[0] = b; new_seq_lens[0] = prefix_len[0] + L + 1;
  }
}

// stage 0: tmp[i][b] = buf_b[src[i]] (if src != dst); stage 1: buf_b[dst[i]] = tmp[i][b]
__global__ void kv_move_kernel(const long* ptrs, int nbuf, int row_int4, const long* src, const long* dst, int N, int4* tmp, int stage) {
  int i = blockIdx.x, b = blockIdx.y;
  long s = src[i], d = dst[i];
  if (s == d) return;
  int4* base = (int4*)ptrs[b];
  int4* t = tmp + ((size_t)i * nbuf + b) * row_int4;
  for (int k = threadIdx.x; k < row_int4; k += blockDim.x) {
    if (stage == 0) t[k] = base[(size_t)s * row_int4 + k];
    else base[(size_t)d * row_int4 + k] = t[k];
  }
}

void tree_build(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, torch::Tensor prefix_len, int64_t D,
                int64_t N, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor lane, torch::Tensor tmask, torch::Tensor fmask,
                torch::Tensor positions) {
  tree_build_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      table.data_ptr<float>(), cand.data_ptr<long>(), anchor.data_ptr<long>(), prefix_len.data_ptr<long>(), (int)D, (int)N,
      tokens.data_ptr<long>(), parent.data_ptr<int>(), depth.data_ptr<int>(), lane.data_ptr<int>(), (unsigned char*)tmask.data_ptr(), (unsigned char*)fmask.data_ptr(),
      positions.data_ptr<long>());
}
void tree_accept(torch::Tensor argmax, torch::Tensor tokens, torch::Tensor parent, int64_t N, torch::Tensor loc,
                 torch::Tensor prefix_len, torch::Tensor out_tokens, torch::Tensor accept_len, torch::Tensor commit_lens, torch::Tensor bonus,
                 torch::Tensor new_seq_lens, torch::Tensor path, torch::Tensor src_slot, torch::Tensor dst_slot) {
  tree_accept_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      argmax.data_ptr<long>(), tokens.data_ptr<long>(), parent.data_ptr<int>(), (int)N, loc.data_ptr<long>(),
      prefix_len.data_ptr<long>(), out_tokens.data_ptr<long>(), accept_len.data_ptr<int>(), commit_lens.data_ptr<int>(), bonus.data_ptr<long>(),
      new_seq_lens.data_ptr<long>(), path.data_ptr<long>(), src_slot.data_ptr<long>(), dst_slot.data_ptr<long>());
}
void kv_move(torch::Tensor ptrs, int64_t row_bytes, torch::Tensor src, torch::Tensor dst, int64_t N, torch::Tensor tmp) {
  int nbuf = ptrs.numel();
  dim3 grid(N, nbuf);
  auto st = at::cuda::getCurrentCUDAStream();
  kv_move_kernel<<<grid, 128, 0, st>>>(ptrs.data_ptr<long>(), nbuf, (int)(row_bytes / 16), src.data_ptr<long>(), dst.data_ptr<long>(), (int)N, (int4*)tmp.data_ptr(), 0);
  kv_move_kernel<<<grid, 128, 0, st>>>(ptrs.data_ptr<long>(), nbuf, (int)(row_bytes / 16), src.data_ptr<long>(), dst.data_ptr<long>(), (int)N, (int4*)tmp.data_ptr(), 1);
}

// ---------------- Phase 4: sampled-draft tree (RRS without replacement) ----------------
// Relabel a best-first tree shape: children of every node v (in index = deterministic-rank order) get lanes drawn WITHOUT replacement from
// q_v = softmax(row_v) by Gumbel-top-k (row_v = table[d][lane_v] at depth d, row 0 for the root). Shape (parent/depth/mask/positions) is
// a function of the table only, so the number of children of each node never depends on its own children's draws.
__global__ void tree_relabel_kernel(const float* table, const long* cand, const int* parent, const int* depth, int* lane, long* tokens,
                                    const float* gum_u, int N, int D) {
  __shared__ float sc[16];
  __shared__ int order[16];
  int t = threadIdx.x;
  for (int v = 0; v < N; ++v) {
    int d = depth[v];
    if (d >= D) continue;
    // any children?
    __syncthreads();
    if (t < 16) {
      int pl = (d == 0) ? 0 : lane[v];
      float x = table[((size_t)d * 16 + pl) * 16 + t];
      float u = gum_u[(size_t)v * 16 + t];
      u = fminf(fmaxf(u, 1e-12f), 1.f - 1e-7f);
      sc[t] = x - logf(-logf(u));
    }
    __syncthreads();
    if (t < 16) {
      int rk = 0;
      for (int k = 0; k < 16; ++k) if (sc[k] > sc[t] || (sc[k] == sc[t] && k < t)) ++rk;
      order[rk] = t;
    }
    __syncthreads();
    if (t == 0) {
      int r = 0;
      for (int j = v + 1; j < N; ++j) if (parent[j] == v) {
        int l = order[r++];
        lane[j] = l; tokens[j] = cand[(size_t)d * 16 + l];
      }
    }
    __syncthreads();
  }
}

// Walk the relabelled tree with recursive rejection sampling without replacement (Jeon et al. 2024, RSD): at node v with children
// c_1..c_k (lanes drawn from q_v without replacement, tested in draw order) keep the residual target r (16 candidate entries + the mass R
// outside the candidate set, which q never touches); accept c_i w.p. min(1, (r(c_i)/Z) / q_i(c_i)), q_i = q_v without c_1..c_{i-1};
// on rejection r <- max(r/Z - q_i, 0), R <- R/Z. If all children are rejected (or v is a leaf) the bonus is drawn from the residual
// (leaf: from p_v). probs: [N, V] fp32 target distribution per node.
__global__ void tree_accept_rrs_kernel(const float* probs, const float* table, const long* cand, const long* tokens, const int* parent,
                                       const int* depth, const int* lane, int N, int D, long V, const float* unif,
                                       const long* loc, const long* prefix_len, long* out_tokens, int* accept_len, int* commit_lens,
                                       long* bonus, long* new_seq_lens, long* path) {
  __shared__ float r[16], q[16];
  __shared__ long ctok[16];
  __shared__ int s_cur, s_L, s_done, s_mode;   // mode: 0 = continue, 1 = draw bonus from residual, 2 = draw bonus from full p
  __shared__ float s_R, s_target;
  __shared__ float csum[256];
  __shared__ long s_bonus;
  int t = threadIdx.x, nt = blockDim.x;
  if (t == 0) { s_cur = 0; s_L = 0; s_done = 0; path[0] = 0; }
  __syncthreads();
  int ui = 0;  // uniform cursor (thread-uniform)
  while (!s_done) {
    int cur = s_cur; int d = depth[cur];
    const float* p = probs + (size_t)cur * V;
    bool has_child = false;
    if (d < D) for (int j = cur + 1; j < N; ++j) if (parent[j] == cur) { has_child = true; break; }
    if (!has_child) {
      if (t == 0) { s_mode = 2; s_done = 1; }
      __syncthreads();
      break;
    }
    if (t < 16) {
      int pl = (d == 0) ? 0 : lane[cur];
      const float* row = table + ((size_t)d * 16 + pl) * 16;
      float mx = -1e30f; for (int k = 0; k < 16; ++k) mx = fmaxf(mx, row[k]);
      float den = 0.f; for (int k = 0; k < 16; ++k) den += __expf(row[k] - mx);
      q[t] = __expf(row[t] - mx) / den;
      ctok[t] = cand[(size_t)d * 16 + t];
      r[t] = p[ctok[t]];
    }
    __syncthreads();
    if (t == 0) {
      float sr = 0.f; for (int k = 0; k < 16; ++k) sr += r[k];
      float R = fmaxf(1.f - sr, 0.f);
      float qmass = 1.f;
      int next = -1;
      for (int j = cur + 1; j < N && next < 0; ++j) {
        if (parent[j] != cur) continue;
        int l = lane[j];
        float Z = R; for (int k = 0; k < 16; ++k) Z += r[k];
        float qi = q[l] / qmass;
        float u = unif[ui++];
        if (Z > 0.f && u * qi * Z < r[l]) { next = j; break; }
        // reject: residual update against q_i (q without the already-tried lanes, renormalised)
        for (int k = 0; k < 16; ++k) r[k] = fmaxf(r[k] / Z - q[k] / qmass, 0.f);
        R = R / Z;
        qmass -= q[l]; q[l] = 0.f;
        if (qmass <= 1e-12f) qmass = 1e-12f;
      }
      if (next >= 0) { s_cur = next; s_L++; path[s_L] = next; s_mode = 0; }
      else { s_mode = 1; s_done = 1; s_R = R; }
    }
    __syncthreads();
    // keep uniform cursor in sync for all threads (only thread 0 uses it; broadcast)
    if (t == 0) csum[0] = (float)ui;
    __syncthreads();
    ui = (int)csum[0];
    __syncthreads();
  }
  // ---- bonus draw at node s_cur
  int cur = s_cur;
  const float* p = probs + (size_t)cur * V;
  float u = unif[N * 17 + 0];   // dedicated final uniforms
  float u2 = unif[N * 17 + 1];
  if (s_mode == 1) {
    // residual: candidate part r[], outside part R * p|_{not cand}
    if (t == 0) {
      float sr = 0.f; for (int k = 0; k < 16; ++k) sr += r[k];
      float Z = s_R + sr;
      float x = u * Z; s_bonus = -1;
      if (x < sr) {
        float c = 0.f; int pick = -1;
        for (int k = 0; k < 16; ++k) { if (r[k] > 0.f) { pick = k; c += r[k]; if (x < c) break; } }
        s_bonus = ctok[pick];
      }
    }
    __syncthreads();
  } else if (t == 0) { s_bonus = -1; }
  __syncthreads();
  if (s_bonus < 0) {
    // inverse-CDF draw from p (mode 2) or from p restricted to non-candidates (mode 1), parallel over V
    bool excl = (s_mode == 1);
    long chunk = (V + nt - 1) / nt;
    long b0 = t * chunk, b1 = min(V, b0 + chunk);
    float acc = 0.f;
    for (long i = b0; i < b1; ++i) {
      float v = p[i];
      if (excl) { for (int k = 0; k < 16; ++k) if (ctok[k] == i) { v = 0.f; break; } }
      acc += v;
    }
    csum[t] = acc;
    __syncthreads();
    if (t == 0) {
      float tot = 0.f; for (int k = 0; k < nt; ++k) tot += csum[k];
      float x = u2 * tot; float c = 0.f; int ch = nt - 1;
      for (int k = 0; k < nt; ++k) { if (csum[k] > 0.f) { ch = k; if (x < c + csum[k]) break; } c += csum[k]; }
      // scan inside chunk ch
      long i0 = ch * chunk, i1 = min(V, i0 + chunk); long pick = -1; float cc = c;
      for (long i = i0; i < i1; ++i) {
        float v = p[i];
        if (excl) { for (int k = 0; k < 16; ++k) if (ctok[k] == i) { v = 0.f; break; } }
        if (v > 0.f) { pick = i; cc += v; if (x < cc) break; }
      }
      s_bonus = pick < 0 ? 0 : pick;
    }
    __syncthreads();
  }
  if (t == 0) {
    int L = s_L;
    long b = s_bonus;
    for (int i = 0; i < N; ++i) out_tokens[i] = 0;
    for (int i = 0; i < L; ++i) out_tokens[i] = tokens[path[i + 1]];
    out_tokens[L] = b;
    for (int i = L + 1; i < N; ++i) path[i] = i;
    accept_len[0] = L; commit_lens[0] = L + 1; bonus[0] = b; new_seq_lens[0] = prefix_len[0] + L + 1;
  }
}

__global__ void slots_kernel(const long* loc, const long* path, int N, long* src_slot, long* dst_slot) {
  int i = threadIdx.x;
  if (i < N) { src_slot[i] = loc[path[i]]; dst_slot[i] = loc[i]; }
}

void tree_relabel(torch::Tensor table, torch::Tensor cand, torch::Tensor parent, torch::Tensor depth, torch::Tensor lane, torch::Tensor tokens,
                  torch::Tensor gum_u, int64_t N, int64_t D) {
  tree_relabel_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(table.data_ptr<float>(), cand.data_ptr<long>(), parent.data_ptr<int>(),
      depth.data_ptr<int>(), lane.data_ptr<int>(), tokens.data_ptr<long>(), gum_u.data_ptr<float>(), (int)N, (int)D);
}
void tree_accept_rrs(torch::Tensor probs, torch::Tensor table, torch::Tensor cand, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth,
                     torch::Tensor lane, int64_t N, int64_t D, torch::Tensor unif, torch::Tensor loc, torch::Tensor prefix_len,
                     torch::Tensor out_tokens, torch::Tensor accept_len, torch::Tensor commit_lens, torch::Tensor bonus, torch::Tensor new_seq_lens,
                     torch::Tensor path, torch::Tensor src_slot, torch::Tensor dst_slot) {
  auto st = at::cuda::getCurrentCUDAStream();
  tree_accept_rrs_kernel<<<1, 256, 0, st>>>(probs.data_ptr<float>(), table.data_ptr<float>(), cand.data_ptr<long>(), tokens.data_ptr<long>(),
      parent.data_ptr<int>(), depth.data_ptr<int>(), lane.data_ptr<int>(), (int)N, (int)D, probs.size(1), unif.data_ptr<float>(),
      loc.data_ptr<long>(), prefix_len.data_ptr<long>(), out_tokens.data_ptr<long>(), accept_len.data_ptr<int>(), commit_lens.data_ptr<int>(),
      bonus.data_ptr<long>(), new_seq_lens.data_ptr<long>(), path.data_ptr<long>());
  slots_kernel<<<1, 128, 0, st>>>(loc.data_ptr<long>(), path.data_ptr<long>(), (int)N, src_slot.data_ptr<long>(), dst_slot.data_ptr<long>());
}
'''
CPP_SRC = r'''
void tree_build(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, torch::Tensor prefix_len, int64_t D,
                int64_t N, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor lane, torch::Tensor tmask, torch::Tensor fmask,
                torch::Tensor positions);
void tree_accept(torch::Tensor argmax, torch::Tensor tokens, torch::Tensor parent, int64_t N, torch::Tensor loc,
                 torch::Tensor prefix_len, torch::Tensor out_tokens, torch::Tensor accept_len, torch::Tensor commit_lens, torch::Tensor bonus,
                 torch::Tensor new_seq_lens, torch::Tensor path, torch::Tensor src_slot, torch::Tensor dst_slot);
void kv_move(torch::Tensor ptrs, int64_t row_bytes, torch::Tensor src, torch::Tensor dst, int64_t N, torch::Tensor tmp);
void tree_relabel(torch::Tensor table, torch::Tensor cand, torch::Tensor parent, torch::Tensor depth, torch::Tensor lane, torch::Tensor tokens,
                  torch::Tensor gum_u, int64_t N, int64_t D);
void tree_accept_rrs(torch::Tensor probs, torch::Tensor table, torch::Tensor cand, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth,
                     torch::Tensor lane, int64_t N, int64_t D, torch::Tensor unif, torch::Tensor loc, torch::Tensor prefix_len,
                     torch::Tensor out_tokens, torch::Tensor accept_len, torch::Tensor commit_lens, torch::Tensor bonus, torch::Tensor new_seq_lens,
                     torch::Tensor path, torch::Tensor src_slot, torch::Tensor dst_slot);
'''
_mod = None
def get_module():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _mod = load_inline("rdf2_tree_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["tree_build", "tree_accept", "kv_move", "tree_relabel", "tree_accept_rrs"],
                           with_cuda=True, extra_cuda_cflags=["-O3"], verbose=False)
    return _mod

class TreeVerifier:
    """Best-first tree of N nodes (anchor + N-1 highest joint-probability nodes, depth <= D) built per round on the GPU."""
    def __init__(self, N, D, device="cuda", max_ctx=8192):
        self.m = get_module()
        self.D = D; self.N = N; self.device = device
        assert N <= 128
        self.lane = torch.zeros(N, dtype=torch.int32, device=device)
        self.tokens = torch.zeros(N, dtype=torch.int64, device=device)
        self.parent = torch.zeros(N, dtype=torch.int32, device=device)
        self.depth = torch.zeros(N, dtype=torch.int32, device=device)
        self.tmask = torch.zeros(N * N, dtype=torch.uint8, device=device)
        self.fmask = None  # packed-bit mask buffer = the attention backend's cuda_graph_custom_mask (set via set_mask_buffer)
        self.positions = torch.zeros(N, dtype=torch.int64, device=device)
        self.out_tokens = torch.zeros(N, dtype=torch.int64, device=device)
        self.accept_len = torch.zeros(1, dtype=torch.int32, device=device)
        self.commit_lens = torch.zeros(1, dtype=torch.int32, device=device)
        self.bonus = torch.zeros(1, dtype=torch.int64, device=device)
        self.new_seq_lens = torch.zeros(1, dtype=torch.int64, device=device)
        self.path = torch.zeros(N, dtype=torch.int64, device=device)
        self.src_slot = torch.zeros(N, dtype=torch.int64, device=device)
        self.dst_slot = torch.zeros(N, dtype=torch.int64, device=device)
        self.kv_ptrs = None; self.kv_tmp = None; self.row_bytes = 0

    def build(self, table, cand, anchor, prefix_len):
        """table [>=D,16,16] f32, cand [>=D,16] int64, anchor [1] int64, prefix_len [1] int64 (device)."""
        self.m.tree_build(table.contiguous(), cand.contiguous(), anchor, prefix_len, self.D, self.N, self.tokens, self.parent,
                          self.depth, self.lane, self.tmask, self.fmask, self.positions)

    def relabel(self, table, cand, gum_u):
        """Phase 4 sampled drafts: keep the built shape, redraw every node's children lanes without replacement from q (Gumbel-top-k).
        table must be the same (temperature-scaled) table later passed to accept_rrs; gum_u [N*16] fresh uniforms."""
        self.m.tree_relabel(table.contiguous(), cand.contiguous(), self.parent, self.depth, self.lane, self.tokens, gum_u, self.N, self.D)

    def accept_rrs(self, probs, table, cand, unif, loc, prefix_len):
        """probs [N, V] fp32 target per node, unif [N*17+2] fresh uniforms. Recursive rejection sampling without replacement."""
        self.m.tree_accept_rrs(probs, table.contiguous(), cand.contiguous(), self.tokens, self.parent, self.depth, self.lane, self.N, self.D, unif,
                               loc, prefix_len, self.out_tokens, self.accept_len, self.commit_lens, self.bonus, self.new_seq_lens, self.path,
                               self.src_slot, self.dst_slot)
        if self.kv_ptrs is not None:
            self.m.kv_move(self.kv_ptrs, self.row_bytes, self.src_slot, self.dst_slot, self.N, self.kv_tmp)

    def set_mask_buffer(self, buf):
        assert buf.dtype == torch.uint8 and buf.is_contiguous()
        self.fmask = buf

    def set_kv(self, k_buffers, v_buffers):
        bufs = list(k_buffers) + list(v_buffers)
        b0 = bufs[0]
        assert b0.dim() == 3 and all(b.is_contiguous() and b.shape == b0.shape for b in bufs), b0.shape
        self.row_bytes = b0.shape[1] * b0.shape[2] * b0.element_size()
        assert self.row_bytes % 16 == 0
        self.kv_ptrs = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=self.device)
        self.kv_tmp = torch.empty(self.N * len(bufs) * self.row_bytes, dtype=torch.uint8, device=self.device)
        self.bufs = bufs

    def accept(self, argmax, loc, prefix_len):
        """argmax [N] int64 (target greedy per node), loc [N] int64 physical slots of the verify window, prefix_len [1] int64."""
        self.m.tree_accept(argmax, self.tokens, self.parent, self.N, loc, prefix_len, self.out_tokens, self.accept_len,
                           self.commit_lens, self.bonus, self.new_seq_lens, self.path, self.src_slot, self.dst_slot)
        if self.kv_ptrs is not None:
            self.m.kv_move(self.kv_ptrs, self.row_bytes, self.src_slot, self.dst_slot, self.N, self.kv_tmp)
