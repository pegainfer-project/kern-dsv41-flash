#!/usr/bin/env python3
"""Build every module in kernels.toml into build/<module>.cubin and the MegaMoE
layouts into build/moe_layout_<experts>_sm<sms>.json, then check each against
its pinned sha256.

nvcc runs from the repository root on relative paths and is told the source's
path relative to it, so a cubin does not depend on where the repository is
cloned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path("build")
UPSTREAM = BUILD / "upstream"
CORE = Path("third_party/FlashMLA/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn")
SUBMODULES = [
    "third_party/DeepGEMM/third-party/cutlass",
    "third_party/FlashMLA/csrc/cutlass",
    "third_party/DeepSelect/csrc/3rdparty/cutlass",
]


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def prepare():
    """The upstream trees the include sets name, rebuilt from the submodules every time."""
    missing = [p for p in SUBMODULES if not (ROOT / p / "include").is_dir()]
    if missing:
        raise SystemExit(f"missing submodules {missing}: run scripts/fetch.sh")
    shutil.rmtree(ROOT / UPSTREAM, ignore_errors=True)
    patched = UPSTREAM / "deepgemm-patched"
    shutil.copytree(ROOT / "third_party/DeepGEMM/deep_gemm/include", ROOT / patched / "include")
    for patch in sorted((ROOT / "patches").glob("deepgemm-*.patch")):
        run("patch", "-s", "-p1", "-d", patched, "-i", patch)
    (ROOT / UPSTREAM / "mega-moe").mkdir(parents=True)
    run("python3", "source/moe/fork_moe.py", "third_party/DeepGEMM", UPSTREAM / "mega-moe/sm100_fp8_fp4_mega_moe.cuh")
    (ROOT / UPSTREAM / "flashmla-fused").mkdir(parents=True)
    run("python3", "source/attention/patch_fused.py", CORE / "kernel.cuh", UPSTREAM / "flashmla-fused/fused_kernel.cuh")


def command(nvcc, cfg, entry, *, cubin, out):
    includes = [f"-I{p}" for p in cfg["includes"][entry["includes"]]] if "includes" in entry else []
    # cicc hashes the source's absolute path into every internal-linkage
    # symbol (`_INTERNAL_<hash>_`); naming it relative keeps the bytes the
    # same wherever the repository is cloned.
    device = [f"-arch={cfg['toolchain']['arch']}", "-cubin", "-Xcicc", f"--orig_src_path_name,{entry['source']}"]
    return [nvcc, *cfg["flags"][entry["flags"]], *(device if cubin else []), *includes, "-o", str(out), entry["source"]]


def compile_module(nvcc, cfg, name):
    out = BUILD / f"{name}.cubin"
    run(*command(nvcc, cfg, cfg["kernels"][name], cubin=True, out=out))
    return name, sha256(out)


def layouts(nvcc, cfg):
    spec = cfg["layouts"]
    tool = BUILD / "moe_layout"
    run(*command(nvcc, cfg, {"source": spec["tool"], **spec}, cubin=False, out=tool))
    for experts in spec["experts"]:
        for sms in spec["sms"]:
            out = ROOT / BUILD / f"moe_layout_{experts}_sm{sms}.json"
            out.write_text(subprocess.run([ROOT / tool, str(experts), str(sms)], check=True,
                                          capture_output=True, text=True).stdout)


def pin(shas):
    """Rewrite kernels.toml's sha256 lines to what was just built."""
    path = ROOT / "kernels.toml"
    text = path.read_text()
    for name, sha in shas.items():
        block = re.compile(rf'(\[kernels\.{re.escape(name)}\]\n(?:[^\[\n].*\n)*?sha256 = ")[0-9a-f]*(")')
        text, n = block.subn(rf"\g<1>{sha}\g<2>", text)
        assert n == 1, name
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="*", help="build only these modules (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--pin", action="store_true", help="write the built shas into kernels.toml")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT / "kernels.toml").read_text())
    names = args.modules or list(cfg["kernels"])
    unknown = set(names) - set(cfg["kernels"])
    if unknown:
        raise SystemExit(f"not in kernels.toml: {sorted(unknown)}")
    nvcc = os.environ.get("NVCC", "nvcc")
    version = re.search(r"V(\d+\.\d+\.\d+)", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout)
    want = cfg["toolchain"]["nvcc"]
    print(f"nvcc {version and version[1]} (kernels.toml pins {want})", flush=True)
    (ROOT / BUILD).mkdir(exist_ok=True)
    prepare()
    with ThreadPoolExecutor(args.jobs) as pool:
        built = dict(pool.map(lambda n: compile_module(nvcc, cfg, n), names))
    if not args.modules:
        layouts(nvcc, cfg)
    if args.pin:
        pin(built)
    mismatched = [n for n, sha in built.items() if not args.pin and sha != cfg["kernels"][n]["sha256"]]
    for n, sha in built.items():
        print(f"{n}: {sha}{'  (differs from kernels.toml)' if n in mismatched else ''}")
    if mismatched:
        raise SystemExit(f"{len(mismatched)} of {len(built)} modules differ from kernels.toml; "
                         f"nvcc must be {want}")


if __name__ == "__main__":
    main()
