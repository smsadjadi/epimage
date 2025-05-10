from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def fsl_preproc(fmri_path, out_dir):
    LOGGER.info("Running dummy FSL preprocessing")
    out = pathlib.Path(out_dir)/'preproc.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    return out
