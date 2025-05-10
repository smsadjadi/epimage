from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def seed_correlation(fmri_path, seed_mask):
    LOGGER.info("Computing dummy seed correlation")
    return np.eye(5)
