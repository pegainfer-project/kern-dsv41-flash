#!/usr/bin/env python3
"""kernels.toml is whole without a GPU: every module names a source that exists, flag and include
sets that exist, and a pinned sha256; every include path lies in the repository; every submodule's
gitlink is the commit kernels.toml records."""
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = tomllib.loads((ROOT / "kernels.toml").read_text())
    errors = []
    for name, k in cfg["kernels"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", k["sha256"]):
            errors.append(f"{name}: no pinned sha256")
        if not (ROOT / k["source"]).is_file():
            errors.append(f"{name}: missing source {k['source']}")
        if k["flags"] not in cfg["flags"]:
            errors.append(f"{name}: unknown flag set {k['flags']}")
        if "includes" in k and k["includes"] not in cfg["includes"]:
            errors.append(f"{name}: unknown include set {k['includes']}")
        if not set(k["programs"]) <= {"load", "prefill"}:
            errors.append(f"{name}: programs {k['programs']}")
    for set_name, paths in cfg["includes"].items():
        errors += [f"include set {set_name}: {p} is outside third_party/, source/ and build/"
                   for p in paths if p.split("/")[0] not in ("third_party", "source", "build")]
    tree = subprocess.run(["git", "ls-tree", "HEAD", "third_party/"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    links = {line.split("\t")[1].removeprefix("third_party/"): line.split()[2] for line in tree.splitlines()}
    for name, up in cfg["upstreams"].items():
        if links.get(name) != up["commit"]:
            errors.append(f"third_party/{name} is at {links.get(name)}, kernels.toml records {up['commit']}")
    errors += [f"third_party/{n} is in no [upstreams] entry" for n in sorted(set(links) - set(cfg["upstreams"]))]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"kernels.toml: {len(cfg['kernels'])} modules, {len(cfg['upstreams'])} upstreams, OK")


if __name__ == "__main__":
    main()
