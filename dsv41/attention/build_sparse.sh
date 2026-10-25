#!/usr/bin/env bash
set -euo pipefail
out=${1:?artifacts}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
deepgemm=$root/third_party/DeepGEMM
mkdir -p "$out"
common=(-std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -gencode arch=compute_103a,code=sm_103a -I"$deepgemm/deep_gemm/include" -I"$deepgemm/third-party/cutlass/include")
nvcc "${common[@]}" -cubin "$root/source/attention/sparse_indexer.cu" -o "$out/sparse_indexer.cubin"
nvcc "${common[@]}" "$here/sparse_layout.cu" -o "$out/sparse_layout"
"$out/sparse_layout" > "$out/sparse_layout.json"
