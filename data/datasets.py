from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def fetch_dataset(name: str, dest: str|pathlib.Path):
    LOGGER.info("Fetching dataset %s to %s (dummy)", name, dest)
    return pathlib.Path(dest)/name
