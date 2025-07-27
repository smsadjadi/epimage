import os
import mne
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import plotting
from openneuro import download
from utils import setup_logging

from actmap import (
    PipelineConfig,
    load_raw,
    load_electrodes,
    set_montage_and_types,
    make_subject_id,
    build_forward_model,
    create_evoked_or_epochs,
    estimate_sources,
    save_nifti,
)

# Config -------------------------------------------------------------------------
config_path = config_file = Path(__file__).resolve().parent / "config" / "config.yml"
with config_path.open("r", encoding="utf-8") as f: config = yaml.safe_load(f)
dataset_dir = Path(config['paths']['ieeg_dir'])
dataset_id = config['paths']['dataset_id']
subjects_dir = dataset_dir / dataset_id
sub = config['paths']['sub']
ses = config['paths']['ses']

# Logger -------------------------------------------------------------------------
# log_file = Path(subjects_dir) / sub / "ieeg_sources.log"
# setup_logging(log_file)

# Download sample data -----------------------------------------------------------
os.makedirs(dataset_dir, exist_ok=True)
# %cd $dataset_dir
# ! openneuro-py download --dataset=$dataset_id --include=$sub
# %cd Path(os.getcwd())

# Define paths -------------------------------------------------------------------
ieeg_file = next((subjects_dir / sub / ses / "ieeg").glob("*_ieeg.vhdr"))
electrodes_tsv = subjects_dir / sub / ses / "ieeg" / f"{sub}_{ses}_electrodes.tsv"
mri_path = next((subjects_dir / sub / ses / "anat").glob("*_T1w.nii"))
print("iEEG file :", ieeg_file)
print("Electrodes:", electrodes_tsv)
print("T1 MRI    :", mri_path)
mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

# Create MRI, Brain, and BEM -----------------------------------------------------
# ! export SUBJECTS_DIR=$subjects_dir
# ! bet <> mri/brain.mgz
# ! mri_convert $mri_path $subjects_dir/mri/T1.mgz

# Prepare electrodes CSV (x,y,z,name,type) ---------------------------------------
# The pipeline expects **comma‑separated** values with at least `x,y,z`.
csv_path = electrodes_tsv.with_suffix('.csv')
df = pd.read_csv(electrodes_tsv, sep='\t')
df_valid = df[np.isfinite(df['x']) & np.isfinite(df['y']) & np.isfinite(df['z'])]
cols = ['name', 'x', 'y', 'z']
if 'hemisphere' in df_valid.columns: cols.append('hemisphere')
df_valid[cols].to_csv(csv_path, index=False)
print("Saved CSV to", csv_path)

# Create a `PipelineConfig` ------------------------------------------------------
cfg = PipelineConfig(
    ieeg_path=ieeg_file,
    electrodes_path=csv_path,
    mri_path=mri_path,
    subjects_dir=subjects_dir,
    spacing_mm=6.0,
    inverse_method="dSPM",
    snr=3.0,
    time_window=(300, 600),
    coord_units="mm",
    keep_time=False,
    verbose=False,
)

# Load row -----------------------------------------------------------------------
raw = load_raw(cfg)
print(raw)
print("Data shape:", raw.get_data().shape)

# Load Electrodes ----------------------------------------------------------------
coords, names, types = load_electrodes(cfg, n_channels=len(df_valid))
print("coords shape:", coords.shape)
print("first 5 coords (m):\n", coords[:5])
print("sample names:", names[:5])
print("types counts:", pd.Series(types).value_counts().to_dict())

# Set montage and types ----------------------------------------------------------
set_montage_and_types(raw, coords, names, types)
print(">> Montage set. Dig points:", len(raw.info['dig']))
raw.set_channel_types({name: "eeg" for name in names})
raw.pick_channels(names, ordered=True)

# Forward model ------------------------------------------------------------------
subject = make_subject_id(cfg)
print(f"Subject inferred: {subject}")
fwd, src = build_forward_model(cfg, raw, subject)
print(f"Forward solution with {fwd['nsource']} sources and {fwd['nchan']} channels.")

# Create evoked or epochs --------------------------------------------------------
data_obj = create_evoked_or_epochs(raw, cfg)
print(type(data_obj), data_obj)

# Estimate sources ---------------------------------------------------------------
zmap, affine = estimate_sources(cfg)
print("Z‑map shape:", zmap.shape)

# Save ---------------------------------------------------------------------------
nii_path = Path(subjects_dir / sub / "ieeg_sources_z.nii.gz")
save_nifti(zmap, affine, nii_path)
print("Saved to", nii_path)

# Testing ------------------------------------------------------------------------

# Interactive plot ---------------------------------------------------------------
plotting.view_img(nii_path, bg_img=mri_path, threshold=3).open_in_browser()
