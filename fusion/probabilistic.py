from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def bayesian_fuse(maps: list[np.ndarray])->np.ndarray:
    LOGGER.info("Bayesian fusing %d maps", len(maps))
    return np.mean(maps, axis=0)
