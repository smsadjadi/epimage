import numpy as np
import nibabel as nib
from nilearn.image import smooth_img


def vbm(t1_img: nib.Nifti1Image, fwhm: float = 8.0) -> nib.Nifti1Image:
    """Simple voxel-based morphometry using spatial smoothing."""
    smoothed = smooth_img(t1_img, fwhm)
    data = smoothed.get_fdata()
    data -= np.mean(data)
    return nib.Nifti1Image(data, t1_img.affine)
