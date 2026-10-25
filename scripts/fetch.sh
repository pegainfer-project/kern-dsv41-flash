#!/usr/bin/env bash
# The pinned upstream checkouts and the one nested submodule each build reads (CUTLASS), shallow.
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --depth 1 third_party/DeepGEMM third_party/FlashMLA third_party/DeepSelect
git -C third_party/DeepGEMM submodule update --init --depth 1 third-party/cutlass
git -C third_party/FlashMLA submodule update --init --depth 1 csrc/cutlass
git -C third_party/DeepSelect submodule update --init --depth 1 csrc/3rdparty/cutlass
