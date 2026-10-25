#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
deepselect=$root/third_party/DeepSelect
mkdir -p "$out"
nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$deepselect/csrc" -I"$deepselect/csrc/3rdparty/cutlass/include" -I"$deepselect/csrc/3rdparty/kerutils/include" \
  "$root/source/attention/select_probe.cu" -lcuda -o "$out/libdsv41_select.so"
(cd "$out" && cuobjdump -xelf all libdsv41_select.so)
nvcc -std=c++20 -O3 -gencode arch=compute_103a,code=sm_103a -cubin "$root/source/attention/candidate.cu" -o "$out/candidate.cubin"
