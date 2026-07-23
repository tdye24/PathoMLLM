"""Load benchmark manifest / run config, expand env vars in sample JSON, subsample."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
SCORERS = {"mcq", "bcnb", "roi_cls", "pathmmu"}


def _resolve(path: str, base: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (base / p).resolve())


def _subst_env(s: str, env: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in env:
            raise KeyError(f"Unset env var: {key}")
        return env[key]

    return re.sub(r"\$\{([^}]+)\}", repl, s)


def _subst_env_deep(obj: Any, env: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _subst_env(obj, env) if "${" in obj else obj
    if isinstance(obj, list):
        return [_subst_env_deep(x, env) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst_env_deep(v, env) for k, v in obj.items()}
    return obj


def load_manifest(path: str | Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("manifest needs a non-empty datasets list")

    for i, entry in enumerate(datasets):
        for key in ("name", "path", "scorer"):
            if not entry.get(key):
                raise ValueError(f"datasets[{i}] missing '{key}'")
        if entry["scorer"] not in SCORERS:
            raise ValueError(f"datasets[{i}] unknown scorer {entry['scorer']!r}")
        p = str(entry["path"])
        if "${" in p:
            p = _subst_env(p, env)
        entry["path"] = _resolve(p, EVAL_DIR)

    if manifest.get("output_dir"):
        manifest["output_dir"] = _resolve(str(manifest["output_dir"]), EVAL_DIR)
    return manifest


def load_run_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("run_name", "inference_script", "model_id", "checkpoint_arg", "checkpoints"):
        if not cfg.get(key):
            raise ValueError(f"run_config missing '{key}'")
    if not cfg["checkpoints"]:
        raise ValueError("run_config checkpoints must be non-empty")
    cfg.setdefault("extra_args", {})
    cfg["inference_script"] = _resolve(str(cfg["inference_script"]), ROOT)
    return cfg


def get_dataset(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest["datasets"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Dataset '{name}' not in manifest")


def sample_entries(
    entry: dict[str, Any],
    manifest: dict[str, Any],
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    env = dict(os.environ if env is None else env)
    samples = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"need non-empty JSON list at {entry['path']}")
    samples = _subst_env_deep(samples, env)

    ids = [str(s.get("id", f"idx_{i}")) for i, s in enumerate(samples)]
    n = len(ids)
    if (c := entry.get("sample_count") or manifest.get("sample_count")) is not None:
        k = min(int(c), n)
    elif (r := entry.get("sample_ratio") or manifest.get("sample_ratio")) is not None:
        k = max(1, int(n * float(r)))
    else:
        k = n
    if k >= n:
        return samples

    rng = np.random.default_rng(int(manifest.get("seed", 42)))
    pick = {ids[i] for i in rng.permutation(n)[:k]}
    by_id = dict(zip(ids, samples))
    return [by_id[i] for i in sorted(pick)]


def sample_by_dataset(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {e["name"]: sample_entries(e, manifest) for e in manifest["datasets"]}
