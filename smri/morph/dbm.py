from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.ants import Registration, ApplyTransforms

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

reg = MapNode(Registration(),
              iterfield=["moving_image"],
              name="reg")
reg.inputs.fixed_image = subj_src.inputs.t1[0]
reg.inputs.moving_image = subj_src.outputs.t1
reg.inputs.transforms = ["SyN"]
reg.inputs.transform_parameters = [(0.1,)]
reg.inputs.metric = ["CC"]
reg.inputs.metric_weight = [1]
reg.inputs.radius_or_number_of_bins = [2]
reg.inputs.convergence_threshold = [1e-6]
reg.inputs.convergence_window_size = [10]
reg.inputs.number_of_iterations = [[100,70,50,20]]
reg.inputs.smoothing_sigmas = [[4,2,1,0]]
reg.inputs.shrink_factors = [[8,4,2,1]]

jac = MapNode(ApplyTransforms(dimension=3,
                              transform_type="Jacobian",
                              output_image="%s_jac.nii.gz"),
              iterfield=["reference_image", "transforms"],
              name="jac")
jac.inputs.reference_image = subj_src.outputs.t1
jac.inputs.transforms      = reg.outputs.forward_transforms

wf = Workflow(name="dbm", base_dir=f"{dataset}/nipype_work")
wf.connect(subj_src, "t1", reg, "moving_image")
wf.connect(reg, "forward_transforms", jac, "transforms")
wf.connect(subj_src, "t1", jac, "reference_image")
wf.run("MultiProc")


# import numpy as np
# import nibabel as nib
# from nilearn.image import resample_to_img


# def dbm(moving: nib.Nifti1Image, template: nib.Nifti1Image) -> nib.Nifti1Image:
#     """Deformation-based morphometry via simple difference after resampling."""
#     resampled = resample_to_img(moving, template)
#     deformation = resampled.get_fdata() - template.get_fdata()
#     return nib.Nifti1Image(deformation, template.affine)