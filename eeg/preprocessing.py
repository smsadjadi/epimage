from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def filter_bandpass(sig, low=1.0, high=40.0, fs=256):
    LOGGER.info("Band‑pass filtering dummy EEG")
    return sig
