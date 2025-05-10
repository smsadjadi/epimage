from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def register(src, ref):
    LOGGER.info("Running dummy registration")
    return {'transform':'identity'}
