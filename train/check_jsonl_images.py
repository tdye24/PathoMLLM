#!/usr/bin/env python3
"""Check that every image path in a swift JSONL exists on OBS (or locally).

Examples:
  python train/check_jsonl_images.py data/roi_cls_vqa.jsonl
  python train/check_jsonl_images.py data/roi_cls_vqa.jsonl \\
      --workers 64 --missing-out missing_images.txt --limit 1000

Remote ``s3://`` / ``obs://`` paths use ``mox.file.exists``; local paths use
``os.path.exists``. Duplicate paths are checked once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


def _extract_paths(images: Any, *, line_no: int) -> list[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images.strip()] if images.strip() else []
    if not isinstance(images, (list, tuple)):
        raise TypeError(f"line {line_no}: images must be str/list, got {type(images).__name__}")

    out: list[str] = []
    for i, item in enumerate(images):
        if isinstance(item, str):
            p = item.strip()
            if p:
                out.append(p)
        elif isinstance(item, dict):
            p = item.get("path") or item.get("image") or item.get("url")
            if isinstance(p, str) and p.strip():
                out.append(p.strip())
            elif item.get("bytes") is not None:
                continue  # inlined bytes, nothing to check on s3
            else:
                raise ValueError(f"line {line_no}: images[{i}] dict missing path: keys={list(item)}")
        else:
            raise TypeError(f"line {line_no}: images[{i}] unsupported type {type(item).__name__}")
    return out


def iter_image_paths(jsonl_path: Path, *, limit: int = 0) -> tuple[list[str], int, int]:
    """Return (unique_paths_in_order, num_rows, num_path_refs)."""
    seen: set[str] = set()
    unique: list[str] = []
    n_rows = 0
    n_refs = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            if limit > 0 and n_rows > limit:
                n_rows -= 1
                break
            row = json.loads(line)
            paths = _extract_paths(row.get("images"), line_no=line_no)
            n_refs += len(paths)
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            if n_rows % 200_000 == 0:
                print(f"[scan] rows={n_rows:,} unique_paths={len(unique):,}", flush=True)

    return unique, n_rows, n_refs


def _is_remote(path: str) -> bool:
    return path.startswith(("s3://", "obs://", "mox://"))


def make_exists_fn():
    """Build an exists(path) -> bool callable; lazy-import mox only if needed."""
    mox = None

    def exists(path: str) -> bool:
        nonlocal mox
        if _is_remote(path):
            if mox is None:
                import moxing as moxing_mod  # type: ignore[import-untyped]

                mox = moxing_mod
            return bool(mox.file.exists(path))
        return os.path.exists(path)

    return exists


def check_paths(
    paths: Iterable[str],
    *,
    workers: int,
    progress_every: int,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Return (ok, missing, errors[(path, msg)])."""
    path_list = list(paths)
    exists = make_exists_fn()
    ok: list[str] = []
    missing: list[str] = []
    errors: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()

    def _one(p: str) -> tuple[str, str, str | None]:
        try:
            return (p, "ok" if exists(p) else "missing", None)
        except Exception as exc:  # noqa: BLE001
            return (p, "error", str(exc))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_one, p) for p in path_list]
        for fut in as_completed(futures):
            p, status, err = fut.result()
            if status == "ok":
                ok.append(p)
            elif status == "missing":
                missing.append(p)
            else:
                errors.append((p, err or "unknown"))
            done += 1
            if progress_every > 0 and done % progress_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"[check] {done:,}/{len(path_list):,} "
                    f"ok={len(ok):,} missing={len(missing):,} err={len(errors):,} "
                    f"({rate:.1f}/s)",
                    flush=True,
                )

    return ok, missing, errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", type=Path, help="swift JSONL with an images field")
    p.add_argument("--workers", type=int, default=32, help="concurrent exists checks (default: 32)")
    p.add_argument("--limit", type=int, default=0, help="only scan first N non-empty rows (0=all)")
    p.add_argument(
        "--missing-out",
        type=Path,
        default=None,
        help="write missing (+ error) paths here (default: <jsonl>.missing.txt)",
    )
    p.add_argument("--progress-every", type=int, default=500, help="print progress every N checks")
    p.add_argument(
        "--fail-on-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit 1 if any path is missing/error (default: true)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jsonl_path: Path = args.jsonl
    if not jsonl_path.is_file():
        print(f"ERROR: jsonl not found: {jsonl_path}", file=sys.stderr)
        return 2

    missing_out: Path = args.missing_out or jsonl_path.with_suffix(jsonl_path.suffix + ".missing.txt")

    print(f"[scan] reading {jsonl_path}", flush=True)
    t0 = time.time()
    unique, n_rows, n_refs = iter_image_paths(jsonl_path, limit=args.limit)
    print(
        f"[scan] done in {time.time() - t0:.1f}s: "
        f"rows={n_rows:,} image_refs={n_refs:,} unique={len(unique):,}",
        flush=True,
    )
    if not unique:
        print("[check] no image paths found", flush=True)
        return 0

    remote = sum(1 for p in unique if _is_remote(p))
    local = len(unique) - remote
    print(f"[check] unique remote={remote:,} local={local:,} workers={args.workers}", flush=True)

    t1 = time.time()
    ok, missing, errors = check_paths(unique, workers=args.workers, progress_every=args.progress_every)
    elapsed = time.time() - t1

    print(
        f"[done] {elapsed:.1f}s  ok={len(ok):,}  missing={len(missing):,}  error={len(errors):,}",
        flush=True,
    )

    if missing or errors:
        missing_out.parent.mkdir(parents=True, exist_ok=True)
        with missing_out.open("w", encoding="utf-8") as f:
            for p in sorted(missing):
                f.write(f"MISSING\t{p}\n")
            for p, msg in sorted(errors):
                f.write(f"ERROR\t{p}\t{msg}\n")
        print(f"[done] wrote {missing_out}", flush=True)
        # show a few examples
        for p in missing[:10]:
            print(f"  MISSING  {p}", flush=True)
        for p, msg in errors[:10]:
            print(f"  ERROR    {p}  ({msg})", flush=True)
        if len(missing) > 10 or len(errors) > 10:
            print(f"  ... see {missing_out}", flush=True)
        return 1 if args.fail_on_missing else 0

    print("[done] all unique image paths exist", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
