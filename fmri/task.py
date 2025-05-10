from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def glm(fmri_path, design_mat):
    LOGGER.info("Running dummy GLM")
    return {'beta': np.ones(1)}
