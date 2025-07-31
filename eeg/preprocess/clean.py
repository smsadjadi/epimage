from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import numpy as np


def clean(eeg_data, sfreq, low=1.0, high=30.0, normalize=False,
          apply_ica=False, n_components=None, reject_components='auto', zscore_thresh=3.0):
    """
    Preprocess EEG data: bandpass filter, optional normalization, optional ICA + artifact removal.
    
    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
        Raw EEG data.
    sfreq : float
        Sampling frequency in Hz.
    low : float
        Low cutoff frequency for bandpass filter.
    high : float
        High cutoff frequency for bandpass filter.
    normalize : bool
        If True, normalize each channel to max absolute amplitude of 1 post-filter.
    apply_ica : bool
        If True, perform ICA decomposition.
    n_components : int | None
        Number of ICA components. Defaults to n_channels if None.
    reject_components : 'auto' | list of int
        Which ICs to remove. If 'auto', use z-score detection.
    zscore_thresh : float
        Threshold for z-score based artifact IC detection.

    Returns
    -------
    eeg_cleaned : ndarray, shape (n_channels, n_samples)
        Preprocessed (and cleaned) EEG data.
    ica_info : dict | None
        Contains ICA object and removed component indices if ICA was applied, else None.
    """
    # 1) Bandpass filter
    b, a = butter(2, [low / (sfreq / 2), high / (sfreq / 2)], btype='band')
    eeg_filtered = filtfilt(b, a, eeg_data, axis=1)
    
    # 2) Normalize per channel
    if normalize:
        eeg_filtered = eeg_filtered / np.max(np.abs(eeg_filtered), axis=1, keepdims=True)


    ica_info = None
    eeg_cleaned = eeg_filtered.copy()
    
    if apply_ica:
        data = eeg_filtered.T
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask] # keep only rows without any NaN

        # 3) ICA decomposition
        n_ch = eeg_filtered.shape[0]
        n_components = n_components or n_ch
        ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
        
        # sklearn expects (n_samples, n_features)
        sources = ica.fit_transform(data)  # shape (n_samples, n_components)
        
        # 4) Determine components to reject
        if reject_components == 'auto':
            zscores = np.abs((sources - np.mean(sources, axis=0)) / np.std(sources, axis=0))
            reject_idx = list(np.where(np.max(zscores, axis=0) > zscore_thresh)[0])
        else:
            reject_idx = list(reject_components)

        # 5) Zero out artifact components and reconstruct
        sources[:, reject_idx] = 0
        reconstructed = ica.inverse_transform(sources).T  # back to (n_channels, n_samples)
        eeg_cleaned = reconstructed

        ica_info = {
            'ica_object': ica,
            'rejected_components': reject_idx
        }
    
    if ica_info:
        return eeg_cleaned, ica_info
    else:
        return eeg_cleaned