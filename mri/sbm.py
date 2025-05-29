import numpy as np
import nibabel as nib
from scipy.ndimage import grey_dilation, grey_erosion


def sbm(t1_img: nib.Nifti1Image, iterations: int = 1) -> nib.Nifti1Image:
    """Surface-based morphometry approximated by morphological thickness."""
    data = t1_img.get_fdata()
    dilated = grey_dilation(data, size=(3, 3, 3), iterations=iterations)
    eroded = grey_erosion(data, size=(3, 3, 3), iterations=iterations)
    thickness = dilated - eroded
    return nib.Nifti1Image(thickness, t1_img.affine)
