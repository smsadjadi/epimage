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