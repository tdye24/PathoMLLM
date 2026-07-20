"""ms-swift external plugin for PathoMLLM SFT on ModelArts.

Loaded via ``--external_plugins pathomllm_plugin.py``. Top-level code runs at
import time in each training worker (before Trainer / callbacks import).

- Shim ``FSDPModule`` for torch 2.4/2.5 (ms-swift 4.3.x imports it eagerly).
- Relax PIL limits for large pathology tiles / PNG metadata.
- Patch ``swift.template.vision_utils.load_file`` for ``s3://`` / ``obs://``.
"""

from __future__ import annotations

import io
import os
import sys

# Before any moxing/protobuf import (moxing old stubs vs protobuf>=4).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_patched = False
_mox = None
_fsdp_shimmed = False


def ensure_fsdp_module_export() -> None:
    """Make ``from torch.distributed.fsdp import FSDPModule`` work on torch<2.6.

    ms-swift 4.3.x eagerly imports activation_cpu_offload → FSDPModule from the
    public fsdp package. On torch 2.4/2.5 it only lives under ``_composable.fsdp``.
    Injecting the symbol once here survives ms-swift reinstalls (lives in this plugin).
    """
    global _fsdp_shimmed
    if _fsdp_shimmed:
        return
    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401

        _fsdp_shimmed = True
        return
    except ImportError:
        pass
    try:
        from torch.distributed._composable.fsdp import FSDPModule
        import torch.distributed.fsdp as fsdp

        fsdp.FSDPModule = FSDPModule
        _fsdp_shimmed = True
        print(
            "[pathomllm_plugin] shimmed FSDPModule into torch.distributed.fsdp (torch<2.6)",
            flush=True,
        )
    except ImportError as exc:
        print(f"[pathomllm_plugin] FSDPModule shim FAILED: {exc}", file=sys.stderr, flush=True)
        raise


def _get_mox():
    global _mox
    if _mox is not None:
        return _mox
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import moxing as mox  # type: ignore[import-untyped]

    _mox = mox
    return _mox


def _read_bytes(path: str) -> bytes:
    mox = _get_mox()
    with mox.file.File(path, "rb") as f:
        data = f.read()
    if not data:
        raise OSError(f"empty read from remote path: {path}")
    return data


def _is_remote(path: object) -> bool:
    return isinstance(path, str) and path.strip().startswith(("s3://", "obs://"))


def apply_remote_image_patch() -> None:
    """Patch swift vision loaders. Safe to call multiple times."""
    global _patched
    if _patched:
        return

    _get_mox()

    import swift.template.vision_utils as vu
    from PIL import Image

    orig_load_file = vu.load_file
    orig_load_image = vu.load_image

    def load_file(path):
        if _is_remote(path):
            return io.BytesIO(_read_bytes(path.strip()))
        return orig_load_file(path)

    def load_image(image):
        if _is_remote(image):
            img = Image.open(io.BytesIO(_read_bytes(image.strip())))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        return orig_load_image(image)

    vu.load_file = load_file
    vu.load_image = load_image

    try:
        import swift.template.base as base

        if getattr(base, "load_image", None) is not None:
            base.load_image = load_image
    except Exception as exc:  # noqa: BLE001
        print(f"[pathomllm_plugin] warn: could not rebind base.load_image: {exc}", file=sys.stderr, flush=True)

    _patched = True
    print("[pathomllm_plugin] patched swift load_file/load_image for s3://", flush=True)


# Import-time side effects (each torchrun worker).
ensure_fsdp_module_export()

try:
    from PIL import Image, PngImagePlugin

    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100 MiB
    Image.MAX_IMAGE_PIXELS = None
except Exception as exc:  # noqa: BLE001
    print(f"[pathomllm_plugin] PIL relax failed: {exc}", file=sys.stderr, flush=True)

apply_remote_image_patch()
