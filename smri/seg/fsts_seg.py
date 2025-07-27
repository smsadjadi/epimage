from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.base import CommandLine

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

fast = MapNode(CommandLine(),
               iterfield=["args"],
               name="fastsurfer")
fast.inputs.cmd = "fastsurfer"

fast.inputs.args = [
    f"--t1 {t1} --sid {sid} --sd {dataset}/fastsurfer"
    for sid, t1 in zip(subjects, subj_src.inputs.t1)
]

wf = Workflow(name="fastsurfer_seg", base_dir=f"{dataset}/nipype_work")
wf.connect(subj_src, "t1", fast, "args")   # for provenance only
wf.run("MultiProc")