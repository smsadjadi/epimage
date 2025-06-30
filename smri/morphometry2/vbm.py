from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.spm import Normalize12, Smooth
from nipype.pipeline.engine import Node

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

gm_maps = [f"{dataset}/{s}/cat12/mri/mwp1{sid}.nii" for sid, s in zip(subjects, subjects)]

norm = MapNode(Normalize12(apply_to_files=gm_maps),
               iterfield=["apply_to_files"],
               name="norm")
norm.inputs.jobtype = "write"
norm.inputs.template = "/opt/spm12/tpm/TPM.nii"

smo  = MapNode(Smooth(fwhm=[8,8,8]),
               iterfield=["in_files"],
               name="smooth")

wf = Workflow(name="vbm", base_dir=f"{dataset}/nipype_work")
wf.connect(norm, "normalized_files", smo, "in_files")
wf.run("MultiProc")