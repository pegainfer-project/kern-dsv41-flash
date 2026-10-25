// kern's SymBuffer: the peer base table lives in device memory.
//
// Upstream DeepGEMM passes `SymBuffer` by value as a __grid_constant__ kernel
// param: the rank's own slab base plus a 72-entry offset table, all baked on
// the host at launch. kern finishes every launch argument at load, before the
// peers' slabs are imported, so the table cannot be a launch-time constant.
// Instead the kernel takes `const int64_t* bases` — a kern `peer` buffer, one
// device address per rank, filled by `Runtime::import_peers` — and this view
// reads the table on the device. Everything upstream calls on a SymBuffer
// (`get_base_ptr`, `map`, `rank_idx`) keeps its meaning.
//
// This header shadows `deep_gemm/layout/sym_buffer.cuh` by include-path order
// (see build.sh); the impl and comm/barrier.cuh compile against it unchanged.
#pragma once

#include <cstdint>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    const int64_t* bases;
    int64_t base;
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    CUTLASS_DEVICE SymBuffer(const int64_t* bases, const uint32_t& rank_idx)
        : bases(bases), base(__ldg(bases + rank_idx)), rank_idx(rank_idx) {}

    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        if constexpr (kNumRanks == 1)
            return ptr;

        int64_t mapped_ptr = (__ldg(bases + dst_rank_idx) - base) + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout
