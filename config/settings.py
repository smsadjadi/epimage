from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

DEFAULTS = {
    'work_dir': pathlib.Path.home() / 'epimage_work'
}

def get(key: str):
    """Retrieve a setting, falling back to DEFAULTS."""
    return DEFAULTS.get(key)
