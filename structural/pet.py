from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def sipcom(mri_path, pet_path, out_dir):
    LOGGER.info("Running dummy SIPCOM")
    out = pathlib.Path(out_dir)/'sipcom.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    return out
