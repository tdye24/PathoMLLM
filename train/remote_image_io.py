"""Load OBS images via moxing for ms-swift / qwen_vl_utils.

Patches ``fetch_image`` so string paths are read with moxing (same as v1 h5).
Auto-applied via ``sitecustomize`` when ``train/`` is on PYTHONPATH.
"""

import copy
import io

import moxing as mox  # type: ignore[import-untyped]
import qwen_vl_utils.vision_process as vp
from PIL import Image

_patched = False


def load_image_pil(path: str) -> Image.Image:
    with mox.file.File(path, "rb") as f:
        data = f.read()
    with io.BytesIO(data) as bio:
        image = copy.deepcopy(Image.open(bio))
    return image.convert("RGB")


def apply_remote_image_patch() -> None:
    global _patched
    if _patched:
        return

    orig = vp.fetch_image

    def fetch_image(ele, *args, **kwargs):
        image = ele.get("image", ele.get("image_url"))
        if isinstance(image, str):
            ele = {**ele, "image": load_image_pil(image)}
            ele.pop("image_url", None)
        return orig(ele, *args, **kwargs)

    vp.fetch_image = fetch_image
    _patched = True
