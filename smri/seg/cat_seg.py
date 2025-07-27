from nipype import Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.spm import CAT12Segment
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths("/opt/spm12")

dataset   = "../dataset/subj_01"
subjects  = [f"subj_{i:02d}" for i in range(1, 11)]

subj_src  = MapNode(IdentityInterface(fields=["subj_id", "t1"]),
                    iterfield=["subj_id"],
                    name="iter")
subj_src.inputs.subj_id = subjects
subj_src.inputs.t1      = [f"{dataset}/{s}/t1.nii.gz" for s in subjects]

cat = MapNode(CAT12Segment(),
              iterfield=["data"],
              name="cat12")
cat.inputs.data = subj_src.outputs.t1
cat.inputs.gm_output  = [1, 0, 0]
cat.inputs.wm_output  = [1, 0, 0]
cat.inputs.csf_output = [1, 0, 0]
cat.inputs.output_directory = [f"{dataset}/{s}/cat12" for s in subjects]

wf = Workflow(name="cat_seg", base_dir=f"{dataset}/nipype_work")
wf.connect(subj_src, "t1", cat, "data")
wf.run("MultiProc")