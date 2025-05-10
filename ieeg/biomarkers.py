from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def composite_score(features):
    LOGGER.info("Computing dummy composite score")
    return np.zeros(len(features))
