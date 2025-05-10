from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def detect_hfos(data):
    LOGGER.info("Detecting dummy HFOs")
    return [{'chan':0,'rate':0}]
