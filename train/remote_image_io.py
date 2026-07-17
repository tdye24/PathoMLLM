"""Load OBS images via moxing for ms-swift.

ms-swift reads images with ``swift.template.vision_utils.load_file``
(plain ``open(path, 'rb')``), not ``qwen_vl_utils.fetch_image``. Patch
``load_file`` / ``load_image`` so ``s3://`` / ``obs://`` are read via moxing
into a BytesIO (same pattern as v1 h5 loading).
"""

from __future__ import annotations

import io
import sys

_patched = False


def _read_bytes(path: str) -> bytes:
    import moxing as mox  # type: ignore[import-untyped]

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

    # base.py does ``from ...vision_utils import load_image`` — rebind that name too
    try:
        import swift.template.base as base

        if getattr(base, "load_image", None) is not None:
            base.load_image = load_image
    except Exception as exc:  # noqa: BLE001
        print(f"[remote_image_io] warn: could not rebind base.load_image: {exc}", file=sys.stderr, flush=True)

    _patched = True
    print("[remote_image_io] patched swift load_file/load_image for s3://", flush=True)


def ensure_patched() -> None:
    """Apply patch or raise (for preflight checks in sft.sh)."""
    apply_remote_image_patch()
    if not _patched:
        raise RuntimeError("remote_image_io patch did not apply")
