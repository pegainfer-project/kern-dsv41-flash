// Flash target: three expert groups of 128, one instance per run in
// `gate.py` RUNS and per SM count it lands on (GB300 152, B300 148): split-K 8
// over 16-token tiles, then 2-CTA tiles of 176 split twice and 224 unsplit.
// The 4-stage split-K 1 instance streams each group's 3.9 MB weight through
// one CTA (LEGACY). The draft gate fills the device.
#include <deep_gemm/impls/sm100_bf16_mega_gate.cuh>
using namespace deep_gemm;
#define GATE(E,BT,ST,TH,MMA,SK,G,SMS,K) reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,E,BT,ST,TH,MMA,SK,G,SMS,K,1,true,false,false,false,true,false,false,false>)
void instantiate_dsv41_gate() {
 void* gates[]={
  GATE(384,16,4,128,1,1,3,150,6),
  GATE(384,16,12,128,1,8,3,144,6),
  GATE(384,176,11,896,2,2,3,144,6),
  GATE(384,224,10,896,2,1,3,150,6),
  GATE(384,224,10,896,2,1,3,144,6),
  GATE(128,16,4,128,1,1,1,152,3),
  GATE(128,16,4,128,1,1,1,148,3),
 };
 asm volatile(""::"g"(gates));
}
