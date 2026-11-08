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
'''
CPP_SRC = r'''
void tree_build(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, torch::Tensor prefix_len, int64_t D,
                int64_t N, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor lane, torch::Tensor tmask, torch::Tensor fmask,
                torch::Tensor positions);
void tree_accept(torch::Tensor argmax, torch::Tensor tokens, torch::Tensor parent, int64_t N, torch::Tensor loc,
                 torch::Tensor prefix_len, torch::Tensor out_tokens, torch::Tensor accept_len, torch::Tensor commit_lens, torch::Tensor bonus,
                 torch::Tensor new_seq_lens, torch::Tensor path, torch::Tensor src_slot, torch::Tensor dst_slot);
void kv_move(torch::Tensor ptrs, int64_t row_bytes, torch::Tensor src, torch::Tensor dst, int64_t N, torch::Tensor tmp);
'''
_mod = None
def get_module():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _mod = load_inline("rdf2_tree_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["tree_build", "tree_accept", "kv_move"],
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
