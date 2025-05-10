from __future__ import annotations
import logging, pathlib, numpy as np
LOGGER = logging.getLogger(__name__)

def train_fusion_model(features, labels):
    LOGGER.info("Training dummy fusion model")
    class Dummy: 
        def predict(self, X): return np.zeros(len(X))
    return Dummy()
