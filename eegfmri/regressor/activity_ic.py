import numpy as np
from numpy.linalg import svd


def ica(eeg_data: np.ndarray, spike_pattern: np.ndarray, n_components: int = 20) -> np.ndarray:
    """ICA-based regressor correlating spike pattern with components."""
    if spike_pattern.ndim != 1:
        raise ValueError("spike_pattern must be 1D")
    eeg_centered = eeg_data - eeg_data.mean(axis=1, keepdims=True)
    u, s, vt = svd(eeg_centered, full_matrices=False)
    components = vt[:n_components]
    best_corr = np.zeros(components.shape[1])
    best_score = -np.inf
    for comp in components:
        if len(comp) < len(spike_pattern):
            continue
        corr = np.correlate(comp, spike_pattern, mode='valid')
        score = np.max(np.abs(corr))
        if score > best_score:
            best_score = score
            best_corr = corr
    return best_corr
