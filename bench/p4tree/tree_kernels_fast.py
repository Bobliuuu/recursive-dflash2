"""Phase 4: lower-overhead tree kernels (bs=1, greedy). Same best-first tree as bench/p2tree/tree_kernels.py (identical selection order and
tie-breaks), split in two so the expensive part can live inside the DSpine draft CUDA graph:
  tree_select  (capture-safe, no prefix length needed): table [D,16,16] -> tokens/parent/depth + 64-bit ancestor sets. All row softmaxes and
               per-row top-8 lane orders are computed in parallel first; the 47 pops then run in one warp with no block-wide barriers.
  tree_pack    (one launch per round, outside the graph): writes the bit-packed FlashInfer custom mask for prefix length P, verify positions
               P+depth, commit positions P+i and the verify input tokens.
Accept / KV compaction reuse the Phase 2 kernels (tree_kernels.TreeVerifier.accept)."""
import os, torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#define EXPAND 8
#define MAXN 128
#define MAXD 64
#define NW 2   // 64-bit words of ancestor bitset (N <= 128)

__global__ void tree_select_kernel(const float* __restrict__ table, const long* __restrict__ cand, const long* __restrict__ anchor,
                                   int D, int N, long* tokens, int* parent, int* depth, unsigned long long* anc) {
  extern __shared__ float sm[];
  float* prob = sm;                                   // [D*16*16]
  long* cs = (long*)(prob + D * 256);                        // [D*16] candidate ids
  unsigned char* order = (unsigned char*)(cs + D * 16);      // [D*16*EXPAND] lane of rank r
  __shared__ long ntok[MAXN];
  __shared__ float nscore[MAXN];
  __shared__ int nlane[MAXN], ndep[MAXN], npar[MAXN];
  __shared__ unsigned long long nanc[MAXN * NW];
  int t = threadIdx.x, nt = blockDim.x;
  // 1) all row softmaxes (same formula/order as the Phase 2 kernel), one thread per (d, predecessor lane) row
  for (int i = t; i < D * 16; i += nt) cs[i] = cand[i];
  for (int rI = t; rI < D * 16; rI += nt) {
    const float* row = table + (size_t)rI * 16;
    float mx = -1e30f;
    for (int k = 0; k < 16; ++k) mx = fmaxf(mx, row[k]);
    float den = 0.f;
    for (int k = 0; k < 16; ++k) den += __expf(row[k] - mx);
    float p[16];
    for (int k = 0; k < 16; ++k) { p[k] = __expf(row[k] - mx) / den; prob[rI * 16 + k] = p[k]; }
    for (int k = 0; k < 16; ++k) {
      int rk = 0;
      for (int j = 0; j < 16; ++j) if (p[j] > p[k] || (p[j] == p[k] && j < k)) ++rk;
      if (rk < EXPAND) order[rI * EXPAND + rk] = (unsigned char)k;
    }
  }
  __syncthreads();
  if (t >= 32) return;
  // 2) best-first pops in warp 0 only. Children of a node are pushed in rank order with non-increasing joint score, so the best open slot
  //    of each expanded node is its next unconsumed rank ("head"); the global best slot (ties: lowest slot index node*EXPAND+rank) is the best
  //    head (ties: lowest node). Each lane owns nodes t and t+32 (and t+64, t+96 for N > 64).
  if (t == 0) { nscore[0] = 1.f; nlane[0] = 0; ndep[0] = 0; npar[0] = -1; ntok[0] = anchor[0]; }
  for (int w = t; w < NW; w += 32) nanc[w] = (w == 0) ? 1ull : 0ull;
  float hs[4]; int hr[4];          // head score / head rank of my nodes (rank EXPAND = exhausted / not expanded)
  #pragma unroll
  for (int c = 0; c < 4; ++c) { hs[c] = -1.f; hr[c] = EXPAND; }
  __syncwarp();
  for (int it = 0; it < N - 1; ++it) {
    // expand node `it` (owned by lane it%32, slot it/32)
    if ((it & 31) == t) {
      int c = it >> 5, d = ndep[it];
      if (d < D) {
        int rI = d * 16 + ((d == 0) ? 0 : nlane[it]);
        hr[c] = 0; hs[c] = nscore[it] * prob[rI * 16 + order[rI * EXPAND]];
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
    if ((be & 31) == t) {          // owner of the winning head: emit the node and advance the head
      int c = be >> 5, d = ndep[be];
      int rI = d * 16 + ((d == 0) ? 0 : nlane[be]);
      int lane = order[rI * EXPAND + hr[c]];
      npar[node] = be; nlane[node] = lane; ndep[node] = d + 1; nscore[node] = bv;
      ntok[node] = cs[d * 16 + lane];
      for (int w = 0; w < NW; ++w) nanc[node * NW + w] = nanc[be * NW + w] | ((node >> 6) == w ? (1ull << (node & 63)) : 0ull);
      hr[c] += 1;
      hs[c] = hr[c] < EXPAND ? nscore[be] * prob[rI * 16 + order[rI * EXPAND + hr[c]]] : -1.f;
    }
    __syncwarp();
  }
  for (int i = t; i < N; i += 32) { parent[i] = npar[i]; depth[i] = ndep[i]; tokens[i] = ntok[i]; }
  for (int i = t; i < N * NW; i += 32) anc[i] = nanc[i];
}

template <typename PT>
__global__ void tree_pack_kernel(const PT* __restrict__ prefix_len, int N, const int* __restrict__ depth, const unsigned long long* __restrict__ anc,
                                 const long* __restrict__ tokens, unsigned char* fmask, long* positions, long* commit_pos, long* tok_out) {
  long P = (long)prefix_len[0];
  long W = P + N;
  long nbytes = ((long)N * W + 7) / 8;
  int g = blockIdx.x * blockDim.x + threadIdx.x, gs = gridDim.x * blockDim.x;
  for (long j = g; j < nbytes; j += gs) {
    unsigned int byte = 0;
    long idx0 = j * 8;
    long row = idx0 / W, col = idx0 % W;
    for (int b = 0; b < 8; ++b) {
      if (row < N) {
        unsigned int v = (col < P) ? 1u : (unsigned int)((anc[row * NW + ((col - P) >> 6)] >> ((col - P) & 63)) & 1ull);
        byte |= v << b;
      }
      if (++col == W) { col = 0; ++row; }
    }
    fmask[j] = (unsigned char)byte;
  }
  if (g < N) { positions[g] = P + depth[g]; commit_pos[g] = P + g; tok_out[g] = tokens[g]; }
}

void tree_select(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, int64_t D, int64_t N,
                 torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor anc) {
  size_t smem = (size_t)D * 256 * sizeof(float) + (size_t)D * 16 * sizeof(long) + (size_t)D * 16 * EXPAND;
  tree_select_kernel<<<1, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      table.data_ptr<float>(), cand.data_ptr<long>(), anchor.data_ptr<long>(), (int)D, (int)N, tokens.data_ptr<long>(),
      parent.data_ptr<int>(), depth.data_ptr<int>(), (unsigned long long*)anc.data_ptr<int64_t>());
}
void tree_pack(torch::Tensor prefix_len, int64_t N, torch::Tensor depth, torch::Tensor anc, torch::Tensor tokens, torch::Tensor fmask,
               torch::Tensor positions, torch::Tensor commit_pos, torch::Tensor tok_out) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (prefix_len.scalar_type() == torch::kInt64)
    tree_pack_kernel<long><<<16, 256, 0, st>>>(prefix_len.data_ptr<long>(), (int)N, depth.data_ptr<int>(), (unsigned long long*)anc.data_ptr<int64_t>(),
        tokens.data_ptr<long>(), (unsigned char*)fmask.data_ptr(), positions.data_ptr<long>(), commit_pos.data_ptr<long>(), tok_out.data_ptr<long>());
  else
    tree_pack_kernel<int><<<16, 256, 0, st>>>(prefix_len.data_ptr<int>(), (int)N, depth.data_ptr<int>(), (unsigned long long*)anc.data_ptr<int64_t>(),
        tokens.data_ptr<long>(), (unsigned char*)fmask.data_ptr(), positions.data_ptr<long>(), commit_pos.data_ptr<long>(), tok_out.data_ptr<long>());
}
'''
CPP_SRC = r'''
void tree_select(torch::Tensor table, torch::Tensor cand, torch::Tensor anchor, int64_t D, int64_t N,
                 torch::Tensor tokens, torch::Tensor parent, torch::Tensor depth, torch::Tensor anc);
void tree_pack(torch::Tensor prefix_len, int64_t N, torch::Tensor depth, torch::Tensor anc, torch::Tensor tokens, torch::Tensor fmask,
               torch::Tensor positions, torch::Tensor commit_pos, torch::Tensor tok_out);
'''
_mod = None
def get_module():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _mod = load_inline("rdf2_tree_fast_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["tree_select", "tree_pack"],
                           with_cuda=True, extra_cuda_cflags=["-O3"], verbose=False)
    return _mod

class FastTree:
    """Static buffers for one bs=1 tree of N nodes, depth <= D."""
    def __init__(self, N, D, device="cuda"):
        assert N <= 128 and D <= 64
        self.m = get_module(); self.N = N; self.D = D
        i64, i32 = torch.int64, torch.int32
        self.tokens = torch.zeros(N, dtype=i64, device=device)
        self.parent = torch.zeros(N, dtype=i32, device=device)
        self.depth = torch.zeros(N, dtype=i32, device=device)
        self.anc = torch.zeros(N * 2, dtype=i64, device=device)
        self.positions = torch.zeros(N, dtype=i64, device=device)
        self.commit_pos = torch.zeros(N, dtype=i64, device=device)

    def select(self, table, cand, anchor):
        """table [D,16,16] f32 contiguous, cand [D,16] int64, anchor [1] int64 -- capture-safe."""
        self.m.tree_select(table, cand, anchor, self.D, self.N, self.tokens, self.parent, self.depth, self.anc)

    def pack(self, prefix_len, fmask, positions_out, tok_out):
        self.m.tree_pack(prefix_len, self.N, self.depth, self.anc, self.tokens, fmask, positions_out, self.commit_pos, tok_out)
