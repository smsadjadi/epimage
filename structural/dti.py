from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def compute_fa(dwi_path: str|pathlib.Path)->np.ndarray:
    LOGGER.info("Computing dummy FA")
    return np.full((2,2,2), 0.7)
