import numpy as np


def regular(eeg_data: np.ndarray, spike_pattern: np.ndarray) -> np.ndarray:
    """Compute spike-pattern regressor by correlating with raw EEG."""
    if spike_pattern.ndim != 1:
        raise ValueError("spike_pattern must be 1D")
    correlations = []
    for t in range(eeg_data.shape[1] - len(spike_pattern) + 1):
        segment = eeg_data[:, t:t + len(spike_pattern)]
        corr = np.corrcoef(segment.reshape(-1), np.tile(spike_pattern, eeg_data.shape[0]))[0, 1]
        correlations.append(corr)
    return np.array(correlations)
