from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def predict_outcome(features):
    LOGGER.info("Predicting dummy outcome")
    return 0.5
