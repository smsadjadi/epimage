from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def deep_solver(eeg):
    LOGGER.info("Running dummy deep inverse model")
    return np.zeros((100,1))
