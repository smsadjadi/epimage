from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def run_vbm(img_path: str|pathlib.Path, out_dir: str|pathlib.Path)->pathlib.Path:
    """Placeholder VBM returning output file path."""
    out = pathlib.Path(out_dir)/'vbm_map.nii.gz'
    LOGGER.info("Pretending to compute VBM on %s → %s", img_path, out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.touch()
    return out
