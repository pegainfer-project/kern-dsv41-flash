// Instantiate the unmodified DeepGEMM PR #432 shifted mHC kernel, with the
// normalized output stored both as BF16 and as MXFP8: column-major GEMM scales
// for the attention projections (SF_BLOCK_M 0), row-major routed scales plus
// the Mega MoE shared-expert scale pages for the FFN, paged by the block M of
// the MegaMoE instance that reads them (SF_BLOCK_M 16..240).
#include <deep_gemm/impls/sm100_mega_mhc.cuh>
using namespace deep_gemm;
// 16 splits is upstream's deterministic-mode default and its pick from eight
// 64-row m-blocks on (prefill); 40 is what its heuristic picks for up to three
// 64-row m-blocks on 152 SMs (two K-blocks per task), the serving case at
// capacity 128. The SM count is a template
// parameter the split reduction's barriers count on: an instance launched
// on a device with fewer SMs waits for CTAs that never become resident, so
// every device the manifest targets gets its own (GB300 152, B300 148).
template<uint32_t S>void instantiate_dsv41_mhc() {
  auto gemm16 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,0>);
  auto moe16 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,16>);
  auto gemm40 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,0>);
  auto moe40 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,16>);
  auto moe16_32 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,32>);
  auto moe16_64 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,64>);
  auto moe16_128 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,128>);
  auto moe16_192 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,192>);
  auto moe16_240 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,S,true,true,true,240>);
  auto moe40_32 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,32>);
  auto moe40_64 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,64>);
  auto moe40_128 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,128>);
  auto moe40_192 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,192>);
  auto moe40_240 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,S,true,true,true,240>);
  asm volatile("" : : "g"(gemm16), "g"(moe16), "g"(gemm40), "g"(moe40), "g"(moe40_32),
                     "g"(moe40_64), "g"(moe40_128), "g"(moe40_192), "g"(moe40_240), "g"(moe16_32),
                     "g"(moe16_64), "g"(moe16_128), "g"(moe16_192), "g"(moe16_240));
}
template void instantiate_dsv41_mhc<152>();
template void instantiate_dsv41_mhc<148>();
