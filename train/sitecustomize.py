"""Imported via `import sitecustomize` when train/ is on PYTHONPATH.

Pathology images often exceed PIL defaults:
- PNG iCCP/ICC metadata: MAX_TEXT_CHUNK too small
- Large WSI / tile JPEGs: MAX_IMAGE_PIXELS (~179M) decompression bomb check

ms-swift resizes after load; PIL must decode first.

Also patches ``swift.template.vision_utils.load_file`` so jsonl ``images``
(OBS ``s3://``) are loaded via moxing into memory (same pattern as v1 h5).
"""
try:
    from PIL import Image, PngImagePlugin

    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100 MiB
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass

try:
    from remote_image_io import apply_remote_image_patch

    apply_remote_image_patch()
except Exception:
    pass
