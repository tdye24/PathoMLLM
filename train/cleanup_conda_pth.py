#!/usr/bin/env python3
"""Remove pathomllm_remote_image.pth from the active conda env (one-time cleanup).

Older sft.sh versions wrote a .pth into site-packages, which made *every* Python
process in that env try to import a removed module (pip, notebook, unrelated jobs).

Usage:
  conda activate /home/ma-user/envs/qwen35
  python /path/to/PathoMLLM/train/cleanup_conda_pth.py
"""

from __future__ import annotations

import pathlib
import site
import sys

NAME = "pathomllm_remote_image.pth"


def candidate_site_packages() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    try:
        roots.extend(pathlib.Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        us = site.getusersitepackages()
        if us:
            roots.append(pathlib.Path(us))
    except Exception:
        pass
    roots.append(
        pathlib.Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    seen: set[str] = set()
    out: list[pathlib.Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def main() -> int:
    removed = 0
    for sp in candidate_site_packages():
        pth = sp / NAME
        if pth.is_file():
            pth.unlink()
            print(f"removed {pth}", flush=True)
            removed += 1
    if removed:
        print(f"done: removed {removed} file(s) from {sys.prefix}", flush=True)
    else:
        print(f"no {NAME} found under {sys.prefix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
