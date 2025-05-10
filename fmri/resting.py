from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def calc_alff(fmri_path):
    LOGGER.info("Calculating dummy ALFF")
    return np.random.rand(10,10,10)
