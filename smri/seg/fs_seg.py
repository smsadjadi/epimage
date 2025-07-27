from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.freesurfer import ReconAll

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

recon = MapNode(ReconAll(),
                iterfield=["subject_id", "T1_files"],
                name="recon")
recon.inputs.subjects_dir = f"{dataset}/freesurfer"
recon.inputs.subject_id   = subjects
recon.inputs.T1_files     = subj_src.outputs.t1
recon.inputs.args         = "-all"

wf = Workflow(name="fs_seg", base_dir=f"{dataset}/nipype_work")
wf.connect(subj_src,  ("subj_id" , recon, "subject_id"))
wf.connect(subj_src,  ("t1"      , recon, "T1_files"))
wf.run("MultiProc")