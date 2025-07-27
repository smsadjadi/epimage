from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.freesurfer import MRIConvert

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

mris = MapNode(MRIConvert(out_type="gii"),
               iterfield=["in_file"],
               name="surf2gii")
mris.inputs.in_file = [
    f"{dataset}/freesurfer/{s}/surf/lh.thickness" for s in subjects
] + [
    f"{dataset}/freesurfer/{s}/surf/rh.thickness" for s in subjects
]

wf = Workflow(name="sbm", base_dir=f"{dataset}/nipype_work")
wf.connect(subj_src, "t1", mris, "in_file")
wf.run("MultiProc")

# import numpy as np
# import nibabel as nib
# from scipy.ndimage import grey_dilation, grey_erosion


# def sbm(t1_img: nib.Nifti1Image, iterations: int = 1) -> nib.Nifti1Image:
#     """Surface-based morphometry approximated by morphological thickness."""
#     data = t1_img.get_fdata()
#     dilated = grey_dilation(data, size=(3, 3, 3), iterations=iterations)
#     eroded = grey_erosion(data, size=(3, 3, 3), iterations=iterations)
#     thickness = dilated - eroded
#     return nib.Nifti1Image(thickness, t1_img.affine)