from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def make_bem(subject_mri):
    LOGGER.info("Generating dummy BEM for %s", subject_mri)
    return {'bem':'dummy'}
