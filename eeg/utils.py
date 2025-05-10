from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def to_montage(eeg, kind='standard_1020'):
    LOGGER.info("Assigning dummy montage %s", kind)
    return eeg
