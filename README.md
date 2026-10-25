# kern-dsv41-flash

DeepSeek V4.1 Flash on the
[kern runtime](https://github.com/pegainfer-project/kern): every module its
manifest launches, built for GB300 (`sm_103a`) from source in one command
and checked byte for byte against the hashes pinned in
[`kernels.toml`](kernels.toml), and the generator that writes the manifest
over them.

kern contains no model. This repository holds the model's kernel sources,
the upstream kernel libraries they instantiate (as submodules, at the
commits the cubins were built from, unmodified), the build that turns them
into the cubins a manifest pins, and [`dsv41/`](dsv41/README.md): the
manifest generator, the operator oracles against the checkpoint's own
`inference/` code, and the GPU harnesses over kern's public API (a Cargo
workspace pinned to the kern commit whose schema the generator writes).

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
- `build/moe_layout_<experts>_sm<sms>_t<tokens>.json`, the MegaMoE ring layouts
  for 128 and 384 experts at 148 and 152 SMs (B300 and GB300), sized for at
  most `tokens` rows a rank (8192, and 32768 for the 384-expert prefill).

nvcc runs from the repository root on relative paths, so the bytes do not
depend on where the repository is cloned. `build.py <module>...` rebuilds a
subset; `--pin` writes the built hashes into `kernels.toml` after a source
change. `scripts/check.py` (CI) checks `kernels.toml` without a GPU: every
module's source, flag and include sets exist, every hash is pinned, every
submodule sits at the commit recorded under `[upstreams]`.

## Modules

| Module | Source | Upstream |
|---|---|---|
| `copy_rows` | [`copy_rows.cu`](source/copy_rows.cu) | — |
| `dsv41_auxiliary` | [`auxiliary/auxiliary.cu`](source/auxiliary/auxiliary.cu) | — |
| `dsv41_engram_peers` | [`auxiliary/engram_peers.cu`](source/auxiliary/engram_peers.cu) | — |
| `dsv41_head` | [`head.cu`](source/head.cu) | — |
| `dsv41_moe_boundary` | [`moe/boundary.cu`](source/moe/boundary.cu) | — |
| `dsv41_mhc` | [`moe/mega_mhc.cu`](source/moe/mega_mhc.cu) | DeepGEMM |
| `dsv41_gate` | [`moe/mega_gate.cu`](source/moe/mega_gate.cu) | DeepGEMM |
| `dsv41_dense` | [`moe/dense.cu`](source/moe/dense.cu) | DeepGEMM |
| `dsv41_mega_moe` | [`moe/mega_moe.cu`](source/moe/mega_moe.cu) | DeepGEMM MegaMoE, forked onto kern's peer table by [`fork_moe.py`](source/moe/fork_moe.py) |
| `dsv41_moe_stage` | [`moe/stage.cu`](source/moe/stage.cu) | — |
| `dsv41_weight_prep` | [`moe/weight_prep.cu`](source/moe/weight_prep.cu) | — |
| `dsv41_fused_prep` | [`moe/fused_prep.cu`](source/moe/fused_prep.cu) | — |
| `dsv41_fused_decode` | [`attention/fused_probe.cu`](source/attention/fused_probe.cu) | FlashMLA's fused sparse attention, [`patch_fused.py`](source/attention/patch_fused.py) |
| `dsv41_paged_indexer` | [`attention/paged_indexer.cu`](source/attention/paged_indexer.cu) | DeepGEMM |
| `dsv41_sparse_indexer` | [`attention/sparse_indexer.cu`](source/attention/sparse_indexer.cu) | DeepGEMM |
| `dsv41_deepselect` | [`attention/select_probe.cu`](source/attention/select_probe.cu) | DeepSelect |
| `dsv41_candidate` | [`attention/candidate.cu`](source/attention/candidate.cu) | — |
| `dsv41_spec_round` | [`spec_round.cu`](source/spec_round.cu) | — |

Module names are the ones the manifest declares. `dsv41_weight_prep` and
`dsv41_fused_prep` run once, in `load`; `dsv41_spec_round` is the device
glue of the five-draft/six-verify `round`; every other module runs in every
forward. `dsv41_fused_decode` is FlashMLA's fused Q RoPE + sparse attention
+ inverse O RoPE + MXFP8 cast, which prefill and decode both launch.
`dsv41_engram_peers` serves only the manifest that shards the Engram tables
into HBM (`--engram device`).

The one module not built here is `dsv41_engram_constants`: the Engram
tokenizer and hash constants are derived from the checkpoint, so
[`auxiliary/engram_constants.py`](source/auxiliary/engram_constants.py)
generates its source from the model and compiles it into `build/`. It needs
torch, tokenizers and a GPU: the RoPE frequencies are derived on CUDA, as the
pinned manifests' were, and a CPU derivation differs in the last bits.

```sh
python3 source/auxiliary/engram_constants.py <checkpoint> <checkpoint>/inference build
```

## Manifest

```sh
python3 dsv41/gen.py --checkpoint <checkpoint> \
  --capacity 2048 --max-seqs 256 --context 1048576 \
  --out model.json --bundle model-cubins
kern verify model.json
```

`gen.py` reads every cubin from `build/` (`--kernels` to point elsewhere);
`--bundle` copies them out by their pinned hashes, the layout a kern target's
`kernels` directory takes. [`dsv41/README.md`](dsv41/README.md) covers the
options and serving it.

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
arithmetic, memory access or control flow. Everything outside
`third_party/` is kern's, Apache-2.0 ([`LICENSE`](LICENSE)).
