import os
import mne
import pandas as pd


FORMAT_READERS = {
    # EEG-oriented formats
    ".vhdr": mne.io.read_raw_brainvision,
    ".set":  mne.io.read_raw_eeglab,
    ".edf":  mne.io.read_raw_edf,
    ".bdf":  mne.io.read_raw_edf,
    ".gdf":  mne.io.read_raw_gdf,
    ".cnt":  mne.io.read_raw_cnt,
    ".egi":  mne.io.read_raw_egi,
    ".lay":  mne.io.read_raw_persyst,
    ".data": mne.io.read_raw_nicolet,
    ".21e":  mne.io.read_raw_nihon,
    # MEG / multimodal formats
    ".fif":  mne.io.read_raw_fif,
    ".sqd":  mne.io.read_raw_kit,
    ".con":  mne.io.read_raw_ctf,
    ".bin":  mne.io.read_raw_artemis123,
    ".nxe":  mne.io.read_raw_eximia,
    ".mat":  mne.io.read_raw_fieldtrip,
}


def load_raw(path, *reader_args, **reader_kwargs):
    ext = os.path.splitext(path.rstrip(os.sep))[1].lower() if os.path.isdir(path) else os.path.splitext(path)[1].lower()
    try: reader = FORMAT_READERS[ext]
    except KeyError: raise ValueError(f"Unsupported raw format (suffix '{ext}') for '{path}'.")
    return reader(path, *reader_args, **reader_kwargs)