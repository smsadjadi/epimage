from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def eeg_triggered_glm(fmri_path, eeg_events):
    LOGGER.info("Running dummy EEG‑triggered GLM")
    return {'tmap': np.zeros((10,10,10))}
