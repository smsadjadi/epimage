import nibabel as nib
from .vbm import vbm
from .dbm import dbm
from .sbm import sbm


def morph(t1_img: nib.Nifti1Image, template: nib.Nifti1Image | None = None):
    """Run all morphometry analyses and return a dict of results."""
    results = {
        'vbm': vbm(t1_img),
        'sbm': sbm(t1_img)
    }
    if template is not None:
        results['dbm'] = dbm(t1_img, template)
    return results
