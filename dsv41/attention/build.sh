#!/usr/bin/env bash
set -euo pipefail
out=${1:?artifact output directory}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
flashmla=$root/third_party/FlashMLA
mkdir -p "$out"
for probe in probe:sparse_prefill paged_probe:paged_decode; do
  nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
    --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
    -I"$flashmla/csrc" -I"$flashmla/csrc/kerutils/include" -I"$flashmla/csrc/cutlass/include" \
    "$here/${probe%%:*}.cu" -lcuda -o "$out/libdsv41_${probe##*:}.so"
  # The second ELF contains the template; the first is CUDA's support module.
  (cd "$out" && cuobjdump -xelf all "libdsv41_${probe##*:}.so")
done
bash "$here/build_fused.sh" "$out"
