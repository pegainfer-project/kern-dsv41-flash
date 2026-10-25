#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
here=$(cd "$(dirname "$0")" && pwd)
deepgemm=$(cd "$here/../.." && pwd)/third_party/DeepGEMM
mkdir -p "$out"
common=(-std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -gencode arch=compute_103a,code=sm_103a -I"$deepgemm/deep_gemm/include" -I"$deepgemm/third-party/cutlass/include")
nvcc "${common[@]}" -cubin "$here/indexer.cu" -o "$out/indexer.cubin"
nvcc "${common[@]}" "$here/indexer_layout.cu" -o "$out/indexer_layout"
"$out/indexer_layout" > "$out/indexer_smem.txt"
