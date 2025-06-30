from nilearn.glm.first_level import spm_hrf
import numpy as np


def conv_hrf(correlation_series, tr, hrf=None):
    
    if hrf is None:
        hrf = spm_hrf(tr)

    regressor = np.convolve(correlation_series, hrf)[:len(correlation_series)]
    return regressor