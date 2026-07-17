"""Load OBS images via moxing for ms-swift.

ms-swift reads images with its own ``swift.template.vision_utils.load_file``
(plain ``open(path, 'rb')``), NOT ``qwen_vl_utils.fetch_image``. So we patch
``load_file`` to read ``s3://`` / ``obs://`` paths through moxing into a BytesIO,
same pattern as v1 h5 loading.

Auto-applied via ``sitecustomize`` when ``train/`` is on PYTHONPATH.
"""

import io

import moxing as mox  # type: ignore[import-untyped]

_patched = False


def _read_bytes(path: str) -> bytes:
    with mox.file.File(path, "rb") as f:
        return f.read()


def apply_remote_image_patch() -> None:
    global _patched
    if _patched:
        return

    import swift.template.vision_utils as vu

    orig_load_file = vu.load_file

    def load_file(path):
        if isinstance(path, str):
            p = path.strip()
            if p.startswith("s3://") or p.startswith("obs://"):
                return io.BytesIO(_read_bytes(p))
        return orig_load_file(path)

    vu.load_file = load_file
    _patched = True
    print("[remote_image_io] patched swift.template.vision_utils.load_file for s3://", flush=True)
