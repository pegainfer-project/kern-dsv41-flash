#include "mega_moe.cuh"

// Every expert on one rank: dispatch and combine stay in the rank's own slab,
// so the peer table is one entry, the slab's own address. A pipeline stage
// owns its slab as scratch, so where a stage starts `dsv41_slab_init` clears
// the slab's signals and the expert stats (upstream clears them once, at
// allocation; every call leaves them clear) and writes that entry.
extern "C" __global__ void dsv41_slab_init(int64_t* base, unsigned char* slab, int words, int* stats, int experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
    reinterpret_cast<unsigned long long*>(slab)[i] = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < experts) stats[i] = 0;
  if (i == 0) *base = reinterpret_cast<int64_t>(slab);
}

BLOCKS(384,6,1,152,8192)
BLOCKS(128,3,1,152,8192)
BLOCKS(384,6,1,148,8192)
BLOCKS(128,3,1,148,8192)
BLOCKS(384,6,1,152,32768)
BLOCKS(384,6,1,148,32768)
