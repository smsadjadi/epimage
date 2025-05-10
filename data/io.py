from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def load_dummy(path: str|pathlib.Path):
    LOGGER.info("Loading %s (dummy)", path)
    return np.zeros((4,4,4))
def save_dummy(arr, path: str|pathlib.Path):
    LOGGER.info("Saving dummy to %s", path)
