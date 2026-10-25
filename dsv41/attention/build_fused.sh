#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
flashmla=$root/third_party/FlashMLA
core=$flashmla/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn
mkdir -p "$out"
python3 "$root/source/attention/patch_fused.py" "$core/kernel.cuh" "$out/fused_kernel.cuh"
for probe in "$root/source/attention/fused_probe.cu:fused_decode" "$here/fused_prefill_probe.cu:fused_prefill"; do
  nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
    --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
    -I"$out" -I"$core" -I"$flashmla/csrc" -I"$flashmla/csrc/kerutils/include" -I"$flashmla/csrc/cutlass/include" \
    "${probe%%:*}" -lcuda -o "$out/libdsv41_${probe##*:}.so"
  (cd "$out" && cuobjdump -xelf all "libdsv41_${probe##*:}.so")
done
