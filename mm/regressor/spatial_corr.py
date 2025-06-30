import numpy as np
from .activity_map import *


def spatial_corr(eeg_data, epileptic_map):
    
    correlations = []
    for t in range(eeg_data.shape[1]):
        frame = eeg_data[:, t]
        corr = np.corrcoef(frame, epileptic_map)[0, 1]
        correlations.append(abs(corr))  # Use absolute value of correlation

    return np.array(correlations)


def spatial_corr_with_topography(eeg_data, epileptic_topography, electrode_positions):

    correlations = []
    for t in range(eeg_data.shape[1]):
        frame_activity = eeg_data[:, t]

        if len(frame_activity) != len(electrode_positions):
            raise ValueError("Mismatch between activity values and electrode positions.")

        frame_topography = generate_topography(frame_activity, electrode_positions, plot=False)

        valid_idx = ~np.isnan(epileptic_topography) & ~np.isnan(frame_topography)
        if np.any(valid_idx):
            corr = np.corrcoef(
                epileptic_topography[valid_idx].flatten(),
                frame_topography[valid_idx].flatten()
            )[0, 1]
            correlations.append(abs(corr))
        else:
            correlations.append(0)

    return np.array(correlations)