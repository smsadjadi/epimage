from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def fragility_matrix(data):
    LOGGER.info("Computing dummy fragility")
    return np.ones((5,5))
