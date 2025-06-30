from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def run_parallel(func, items):
    LOGGER.info("Running dummy parallel")
    return [func(i) for i in items]
