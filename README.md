# kern-dsv41-flash

The kernels DeepSeek V4.1 Flash runs on the
[kern runtime](https://github.com/pegainfer-project/kern): every module its
`load` and `prefill` programs launch, built for GB300 (`sm_103a`) from
source in one command and checked byte for byte against the hashes pinned
in [`kernels.toml`](kernels.toml).

kern contains no model. This repository holds the model's kernel sources,
the upstream kernel libraries they instantiate (as submodules, at the
commits the cubins were built from, unmodified), and the
build that turns them into the cubins a manifest pins.

## Build

```sh
git clone https://github.com/pegainfer-project/kern-dsv41-flash
cd kern-dsv41-flash
scripts/fetch.sh                                  # the three upstreams and their CUTLASS, shallow
NVCC=<cuda-13.1>/bin/nvcc python3 scripts/build.py
```

CUDA 13.1 (`V13.1.115`) reproduces the pinned cubins; on a 144-core host the
whole set builds in about 11 seconds. `build.py` writes

- `build/<module>.cubin` for every module in `kernels.toml`, each checked
  against its `sha256` (a mismatch fails the build and names the modules);
- `build/moe_layout_<experts>_sm<sms>.json`, the MegaMoE ring layouts for
  128 and 384 experts at 148 and 152 SMs (B300 and GB300).

nvcc runs from the repository root on relative paths, so the bytes do not
depend on where the repository is cloned. `build.py <module>...` rebuilds a
subset; `--pin` writes the built hashes into `kernels.toml` after a source
change. `scripts/check.py` (CI) checks `kernels.toml` without a GPU: every
module's source, flag and include sets exist, every hash is pinned, every
submodule sits at the commit recorded under `[upstreams]`.

## Modules

| Module | Source | Upstream | Program |
|---|---|---|---|
| `copy_rows` | [`copy_rows.cu`](source/copy_rows.cu) | — | prefill |
| `dsv41_auxiliary` | [`auxiliary/auxiliary.cu`](source/auxiliary/auxiliary.cu) | — | prefill |
| `dsv41_engram_peers` | [`auxiliary/engram_peers.cu`](source/auxiliary/engram_peers.cu) | — | prefill |
| `dsv41_head` | [`head.cu`](source/head.cu) | — | prefill |
| `dsv41_moe_boundary` | [`moe/boundary.cu`](source/moe/boundary.cu) | — | prefill |
| `dsv41_mhc` | [`moe/mega_mhc.cu`](source/moe/mega_mhc.cu) | DeepGEMM | prefill |
| `dsv41_gate` | [`moe/mega_gate.cu`](source/moe/mega_gate.cu) | DeepGEMM | prefill |
| `dsv41_dense` | [`moe/dense.cu`](source/moe/dense.cu) | DeepGEMM | prefill |
| `dsv41_mega_moe` | [`moe/mega_moe.cu`](source/moe/mega_moe.cu) | DeepGEMM MegaMoE, forked onto kern's peer table by [`fork_moe.py`](source/moe/fork_moe.py) | prefill |
| `dsv41_moe_stage` | [`moe/stage.cu`](source/moe/stage.cu) | — | prefill |
| `dsv41_weight_prep` | [`moe/weight_prep.cu`](source/moe/weight_prep.cu) | — | load |
| `dsv41_fused_prep` | [`moe/fused_prep.cu`](source/moe/fused_prep.cu) | — | load |
| `dsv41_fused_decode` | [`attention/fused_probe.cu`](source/attention/fused_probe.cu) | FlashMLA's fused sparse attention, [`patch_fused.py`](source/attention/patch_fused.py) | prefill |
| `dsv41_paged_indexer` | [`attention/paged_indexer.cu`](source/attention/paged_indexer.cu) | DeepGEMM | prefill |
| `dsv41_sparse_indexer` | [`attention/sparse_indexer.cu`](source/attention/sparse_indexer.cu) | DeepGEMM | prefill |
| `dsv41_deepselect` | [`attention/select_probe.cu`](source/attention/select_probe.cu) | DeepSelect | prefill |
| `dsv41_candidate` | [`attention/candidate.cu`](source/attention/candidate.cu) | — | prefill |

Module names are the ones the manifest declares: `dsv41_fused_decode` is
FlashMLA's fused Q RoPE + sparse attention + inverse O RoPE + MXFP8 cast,
which prefill and decode both launch. `dsv41_engram_peers` serves only the
manifest that shards the Engram tables into HBM (`--engram device`).

The one module not built here is `dsv41_engram_constants`: the Engram
tokenizer and hash constants are derived from the checkpoint, so
[`auxiliary/engram_constants.py`](source/auxiliary/engram_constants.py)
generates its source from the model and compiles it:

```sh
python3 source/auxiliary/engram_constants.py <checkpoint> <checkpoint>/inference build/constants
```

## Upstreams and licenses

| Submodule | Commit | License |
|---|---|---|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `ab69f76` (PR #432) | MIT |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | `4f38f29` (PR #221) | MIT |
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect) | `8e70df7` | MIT |

Each keeps its own license in its tree. `build.py` never modifies a
submodule; its two source transforms write to `build/upstream/`:
[`fork_moe.py`](source/moe/fork_moe.py) adapts MegaMoE's kernel ABI to
kern's device peer table (and refuses a DeepGEMM source it has not
reviewed), and [`patch_fused.py`](source/attention/patch_fused.py) broadens
two static assertions in FlashMLA's fused core for NVCC 13, changing no
arithmetic, memory access or control flow. The sources
under [`source/`](source/) are kern's, Apache-2.0 ([`source/LICENSE`](source/LICENSE)).
