#include "mega_config.cuh"
#include "sm100_fp8_fp4_mega_moe.cuh"
#define DSV41_MEGA_TM_PARAMS                                                                    \
  const __grid_constant__ CUtensorMap t0, const __grid_constant__ CUtensorMap t1,            \
      const __grid_constant__ CUtensorMap t2, const __grid_constant__ CUtensorMap t3,        \
      const __grid_constant__ CUtensorMap t4, const __grid_constant__ CUtensorMap t5,        \
      const __grid_constant__ CUtensorMap t6, const __grid_constant__ CUtensorMap t7,        \
      const __grid_constant__ CUtensorMap t8, const __grid_constant__ CUtensorMap s0,        \
      const __grid_constant__ CUtensorMap s1, const __grid_constant__ CUtensorMap s2,        \
      const __grid_constant__ CUtensorMap s3, const __grid_constant__ CUtensorMap s4,        \
      const __grid_constant__ CUtensorMap s5, const __grid_constant__ CUtensorMap s6,        \
      const __grid_constant__ CUtensorMap s7, const __grid_constant__ CUtensorMap s8
#define DSV41_MEGA_TM_ARGS t0, t1, t2, t3, t4, t5, t6, t7, t8, s0, s1, s2, s3, s4, s5, s6, s7, s8


// One instance per expert count, SM count, token bound and block M: the SM
// count is a template parameter of the dispatch's grid sync, so a cubin built
// for 152 SMs hangs on a 148-SM device waiting for CTAs that never become
// resident, and the token bound sizes the ring. The block M is how many of an
// expert's rows share one load of its weight tiles; the manifest picks it per
// call from the rank's token count, as upstream's heuristic does. The ring and
// slab hold every candidate block M, so the instances of one bound share a slab.
#define WORLD(E,K,S,T,B) \
extern "C" __global__ __launch_bounds__(512,1) void dsv41_mega_moe_e##E##_r4_sm##S##_t##T##_b##B(void* y,int* stats,uint32_t tokens,const int64_t* peers,uint32_t rank,DSV41_MEGA_TM_PARAMS) { \
 using C=dsv41::Config<E,K,4,S,T,B>; \
 deep_gemm::sm100_fp8_fp4_mega_moe_body<T,5120,2304,E,1,K,C::BlockM,128,128,C::StoreM,C::SfM,128,C::Ring,C::SfRing,C::Stages,5120,128,128,256,S,4,10.f,false,cutlass::detail::float_e2m1_unpacksmem_t>(y,stats,tokens,peers,rank,DSV41_MEGA_TM_ARGS); \
}
#define BLOCKS(E,K,S,T) WORLD(E,K,S,T,16) WORLD(E,K,S,T,32) WORLD(E,K,S,T,64) WORLD(E,K,S,T,128) WORLD(E,K,S,T,192) WORLD(E,K,S,T,240)
BLOCKS(384,6,152,8192)
BLOCKS(128,3,152,8192)
BLOCKS(384,6,148,8192)
BLOCKS(128,3,148,8192)
BLOCKS(384,6,152,32768)
BLOCKS(384,6,148,32768)
