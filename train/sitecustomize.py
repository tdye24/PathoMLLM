"""Imported via `import sitecustomize` when train/ is on PYTHONPATH / site .pth.

Pathology images often exceed PIL defaults:
- PNG iCCP/ICC metadata: MAX_TEXT_CHUNK too small
- Large WSI / tile JPEGs: MAX_IMAGE_PIXELS (~179M) decompression bomb check

Also patches swift vision loaders for OBS ``s3://`` via moxing.
"""
import os
import sys

# Before any moxing/protobuf import (moxing old stubs vs protobuf>=4).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from PIL import Image, PngImagePlugin

    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100 MiB
    Image.MAX_IMAGE_PIXELS = None
except Exception as exc:  # noqa: BLE001
    print(f"[sitecustomize] PIL relax failed: {exc}", file=sys.stderr, flush=True)

try:
    from remote_image_io import apply_remote_image_patch

    apply_remote_image_patch()
except Exception as exc:  # noqa: BLE001
    # Do not swallow — without the patch, every s3:// sample fails.
    print(f"[sitecustomize] remote_image_io patch FAILED: {exc}", file=sys.stderr, flush=True)
