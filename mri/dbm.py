import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img


def dbm(moving: nib.Nifti1Image, template: nib.Nifti1Image) -> nib.Nifti1Image:
    """Deformation-based morphometry via simple difference after resampling."""
    resampled = resample_to_img(moving, template)
    deformation = resampled.get_fdata() - template.get_fdata()
    return nib.Nifti1Image(deformation, template.affine)
