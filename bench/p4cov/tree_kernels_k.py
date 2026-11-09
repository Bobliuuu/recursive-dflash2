"""Phase 4 coverage experiment: the Phase 4 best-first tree selection (bench/p4tree/tree_kernels_fast.py) generalised to K candidates
per position (K <= 64, power of two) and up to E children per node. Two capture-safe launches instead of one:
  tree_rows   one warp per (depth, predecessor lane) row: softmax over K, then the top-E (prob, lane) pairs in rank order
              (ties: lower lane first, the Phase 4 rank rule), written to a global scratch [D*K, E].
  tree_pop    one warp: the 47 best-first pops, same "head per expanded node" scheme and tie-breaks as the Phase 4 kernel.
tree_pack is unchanged (the Phase 4 kernel, imported from tree_kernels_fast). For K=16, E=8 the selected tree equals the Phase 4 one
up to the fp32 summation order of the softmax denominator (checked in test_k.py)."""
import os, sys, torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p4tree"))

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#define MAXN 128
#define NW 2
#define MAXE 16

// one warp per row; lane l holds elements l and l+32 (K <= 64)
__global__ void tree_rows_kernel(const float* __restrict__ table, int R, int K, int E, float* __restrict__ pe, unsigned char* __restrict__ le) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (w >= R) return;
  const float* row = table + (size_t)w * K;
  float a = (l < K) ? row[l] : -INFINITY, b = (l + 32 < K) ? row[l + 32] : -INFINITY;
  float mx = fmaxf(a, b);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  float ea = (l < K) ? __expf(a - mx) : 0.f, eb = (l + 32 < K) ? __expf(b - mx) : 0.f;
  float den = ea + eb;
  for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffff, den, o);
  float pa = (l < K) ? ea / den : -1.f, pb = (l + 32 < K) ? eb / den : -1.f;
  for (int r = 0; r < E; ++r) {
    float bv; int bi;
    if (pa >= pb) { bv = pa; bi = l; } else { bv = pb; bi = l + 32; }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, o); int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (l == 0) { pe[(size_t)w * E + r] = bv; le[(size_t)w * E + r] = (unsigned char)bi; }
    if (bi == l) pa = -2.f; else if (bi == l + 32) pb = -2.f;
  }
}

__global__ void tree_pop_kernel(const float* __restrict__ pe, const unsigned char* __restrict__ le, const long* __restrict__ cand,
                                const long* __restrict__ anchor, int D, int N, int K, int E,
                                long* tokens, int* parent, int* depth, unsigned long long* anc) {
  __shared__ long ntok[MAXN];
  __shared__ float nscore[MAXN];
  __shared__ int nlane[MAXN], ndep[MAXN], npar[MAXN];
  __shared__ unsigned long long nanc[MAXN * NW];
  int t = threadIdx.x;
  if (t == 0) { nscore[0] = 1.f; nlane[0] = 0; ndep[0] = 0; npar[0] = -1; ntok[0] = anchor[0]; }
  for (int w = t; w < NW; w += 32) nanc[w] = (w == 0) ? 1ull : 0ull;
  float hs[4]; int hr[4];
  #pragma unroll
  for (int c = 0; c < 4; ++c) { hs[c] = -1.f; hr[c] = E; }
  __syncwarp();
  for (int it = 0; it < N - 1; ++it) {
    if ((it & 31) == t) {
      int c = it >> 5, d = ndep[it];
      if (d < D) {
        int rI = d * K + ((d == 0) ? 0 : nlane[it]);
        hr[c] = 0; hs[c] = nscore[it] * pe[(size_t)rI * E];
      }
    }
    float bv = -2.f; int be = 0x7fffffff;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      int e = t + 32 * c;
      if (e <= it && (hs[c] > bv || (hs[c] == bv && e < be))) { bv = hs[c]; be = e; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, o); int oe = __shfl_xor_sync(0xffffffff, be, o);
      if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
    }
    const int node = it + 1;
    if ((be & 31) == t) {
      int c = be >> 5, d = ndep[be];
      int rI = d * K + ((d == 0) ? 0 : nlane[be]);
      int lane = le[(size_t)rI * E + hr[c]];
      npar[node] = be; nlane[node] = lane; ndep[node] = d + 1; nscore[node] = bv;
      ntok[node] = cand[(size_t)d * K + lane];
      for (int w = 0; w < NW; ++w) nanc[node * NW + w] = nanc[be * NW + w] | ((node >> 6) == w ? (1ull << (node & 63)) : 0ull);
      hr[c] += 1;
      hs[c] = hr[c] < E ? nscore[be] * pe[(size_t)rI * E + hr[c]] : -1.f;
    }
    __syncwarp();
  }
  for (int i = t; i < N; i += 32) { parent[i] = npar[i]; depth[i] = ndep[i]; tokens[i] = ntok[i]; }
  for (int i = t; i < N * NW; i += 32) anc[i] = nanc[i];
}

void tree_select_k(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, int64_t D, int64_t N, int64_t K, int64_t E,
                   torch::Tensor pe, torch::Tensor le, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor anc) {
  auto st = at::cuda::getCurrentCUDAStream();
  int R = (int)(D * K);
  tree_rows_kernel<<<(R + 7) / 8, 256, 0, st>>>(table.data_ptr<float>(), R, (int)K, (int)E, pe.data_ptr<float>(), (unsigned char*)le.data_ptr());
  tree_pop_kernel<<<1, 32, 0, st>>>(pe.data_ptr<float>(), (unsigned char*)le.data_ptr(), cand.data_ptr<long>(), anchor.data_ptr<long>(),
      (int)D, (int)N, (int)K, (int)E, tokens.data_ptr<long>(), parent.data_ptr<int>(), depth.data_ptr<int>(), (unsigned long long*)anc.data_ptr<int64_t>());
}
'''
CPP_SRC = r'''
void tree_select_k(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, int64_t D, int64_t N, int64_t K, int64_t E,
                   torch::Tensor pe, torch::Tensor le, torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor anc);
'''
_mod = None
def get_module():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _mod = load_inline("rdf2_tree_k_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["tree_select_k"],
                           with_cuda=True, extra_cuda_cflags=["-O3"], verbose=False)
    return _mod


from tree_kernels_fast import FastTree


class FastTreeK(FastTree):
    """FastTree (same buffers, same pack) with a K-wide table and E children per node."""
    def __init__(self, N, D, K, E=8, device="cuda"):
        super().__init__(N, D, device)
        assert K <= 64 and 1 <= E <= 16 and E <= K
        self.K, self.E = K, E
        self.mk = get_module()
        self.pe = torch.zeros(D * K * E, dtype=torch.float32, device=device)
        self.le = torch.zeros(D * K * E, dtype=torch.uint8, device=device)

    def select(self, table, cand, anchor):
        """table [D,K,K] f32 contiguous, cand [D,K] int64 contiguous, anchor [1] int64 -- capture-safe."""
        self.mk.tree_select_k(table, cand, anchor, self.D, self.N, self.K, self.E, self.pe, self.le,
                              self.tokens, self.parent, self.depth, self.anc)
