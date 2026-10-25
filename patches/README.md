# Patches

`build.py` applies every `deepgemm-*.patch` here (`patch -p1`) to a copy
of DeepGEMM's `deep_gemm/include` in `build/upstream/deepgemm-patched/`;
the include set `deepgemm-patched` reads that copy. The two patches touch
disjoint files. A build from the unpatched source produces different
cubins, so the hashes in `kernels.toml` are those of the patched build.

| Patch | Upstream | What it changes | Modules |
|---|---|---|---|
| [`deepgemm-pdl-window.patch`](deepgemm-pdl-window.patch) | DeepGEMM `ab69f76` | `sm100_fp8_fp4_gemm_1d1d`: programmatic dependent launch; the dependent grid is released as the producer drains, its own `griddepcontrol.wait` still ordering the data | `dsv41_dense` |
| [`deepgemm-mhc-gate-early-trigger.patch`](deepgemm-mhc-gate-early-trigger.patch) | DeepGEMM `ab69f76` | Mega mHC and Mega Gate issue `griddepcontrol.launch_dependents` right after their grid-dependency wait, so a PDL successor's prologue overlaps the whole kernel | `dsv41_mhc`, `dsv41_gate` |

Two more source transforms run in `build.py` from scripts beside their
consumers: [`fork_moe.py`](../source/moe/fork_moe.py) adapts MegaMoE's
kernel ABI to kern's device peer table (and refuses a DeepGEMM source it
has not reviewed), and [`patch_fused.py`](../source/attention/patch_fused.py)
broadens two static assertions in FlashMLA's fused core for NVCC 13,
changing no arithmetic, memory access or control flow.

The patches are derived from MIT-licensed DeepSeek code and are under the
same license.
