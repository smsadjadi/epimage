import re
import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import mne
from mne.io import read_raw
from mne import make_bem_model, make_bem_solution, make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.transforms import Transform

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _prepare_raw(raw):
    """Ensure *raw* is an mne.io.Raw instance, loading data into memory."""
    if isinstance(raw, str):
        raw = read_raw(raw, preload=True, verbose=False)
    elif not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("`raw` needs to be a path or an mne.io.Raw object")
    if not raw.preload:
        raw.load_data()
    return raw


def _prepare_montage(raw, electrode_locs):
    """Attach electrode positions to *raw* as a DigMontage.

    Parameters
    ----------
    raw : mne.io.Raw
    electrode_locs : array_like | pandas.DataFrame
        * (n_channels × 3) float array **or**
        * DataFrame with columns ('x', 'y', 'z'[, 'name']).
        Units must be **metres** in MRI space (native or MNI).
    """

    if isinstance(electrode_locs, pd.DataFrame):
        if not {"x", "y", "z"}.issubset(electrode_locs.columns):
            raise ValueError("DataFrame must contain columns 'x', 'y', 'z'")
        coords = electrode_locs[["x", "y", "z"]].to_numpy(float)
        names = (
            electrode_locs["name"].to_list()
            if "name" in electrode_locs.columns
            else raw.ch_names
        )
    else:
        coords = np.asarray(electrode_locs, float)
        if coords.shape[1] != 3:
            raise ValueError("electrode_locs must have shape (n, 3)")
        names = raw.ch_names

    montage = mne.channels.make_dig_montage(
        ch_pos={n: c for n, c in zip(names, coords)}, coord_frame="mri"
    )
    raw.set_montage(montage, on_missing="ignore")
    return raw


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def estimate_sources_ieeg(
    raw,
    electrode_locs,
    mri_path=None,
    time_window=None,
    inverse_method="dSPM",
    snr=3.0,
    spacing="10",
    subjects_dir=None,
    verbose=False,
):
    """Estimate a Z‑scored 3‑D source map from iEEG.

    Parameters
    ----------
    raw : str | mne.io.Raw
        iEEG dataset.  If *str*, any format readable by MNE works (EDF, BrainVision, etc.).
    electrode_locs : array | pandas.DataFrame
        Individual contact coordinates in **MRI space** (metres).
    mri_path : str | None
        Path to subject's *T1‑weighted MRI* processed with FreeSurfer.  If *None*,
        the *fsaverage* template is used.
    time_window : tuple | None
        ``(tmin, tmax)`` in seconds to average.  *None* → entire recording.
    inverse_method : {'MNE', 'dSPM', 'sLORETA'}
        Inverse solver (defaults to *dSPM*).
    snr : float
        Signal‑to‑noise ratio.  Regularisation factor λ² = 1 / snr².
    spacing : str
        Grid resolution of the volumetric source space, e.g. ``'10'`` ≈ 10 mm.
    subjects_dir : str | None
        FreeSurfer subjects directory (defaults to ``$SUBJECTS_DIR``).
    verbose : bool
        Forwarded to MNE routines.

    Returns
    -------
    src_map_z : np.ndarray
        3‑D matrix « X × Y × Z » of Z‑scored activity.
    affine : np.ndarray
        Affine mapping voxel indices → MRI RAS coordinates.
    """

    # ------------------------------------------------------------------
    # 0.  I/O & preprocessing
    # ------------------------------------------------------------------
    raw = _prepare_raw(raw)
    raw = _prepare_montage(raw, electrode_locs)

    # ------------------------------------------------------------------
    # 1.  Subject handling / MRI
    # ------------------------------------------------------------------
    if mri_path is None:
        subject = "fsaverage"
    else:
        # Try to infer subject ID from the directory structure
        mri_path = Path(mri_path)
        m = re.search(r"sub-[A-Za-z0-9]+", str(mri_path))
        subject = m.group(0) if m else mri_path.parent.name

    # ------------------------------------------------------------------
    # 2.  Create source space, BEM & forward model
    # ------------------------------------------------------------------
    src = mne.setup_volume_source_space(
        subject=subject,
        pos=float(spacing),
        subjects_dir=subjects_dir,
        add_interpolator=False,
        verbose=verbose,
    )

    bem_model = make_bem_model(
        subject=subject, subjects_dir=subjects_dir, conductivity=(0.3,), verbose=verbose
    )
    bem = make_bem_solution(bem_model)

    # Identity transform → electrodes already expressed in MRI coords
    trans = Transform("mri", "head")

    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=False,
        # mindist=0.0,  # keep defaults unless you need a source-to-inner-skull buffer
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 3.  Prepare data / Evoked
    # ------------------------------------------------------------------
    if time_window is not None:
        tmin, tmax = time_window
        raw_win = raw.copy().crop(tmin, tmax)
    else:
        raw_win = raw

    mean_data = raw_win.get_data().mean(axis=1, keepdims=True)  # (n_channels, 1)
    evoked = mne.EvokedArray(mean_data, raw.info, tmin=0.0)

    # Simple diagonal noise cov – works well for averaged data
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)

    lambda2 = 1.0 / snr ** 2
    inverse_operator = make_inverse_operator(
        evoked.info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose=verbose,
    )

    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=inverse_method.lower(),
        pick_ori=None,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 4.  Convert to volume + Z‑score
    # ------------------------------------------------------------------
    img = stc.as_volume(src, mri_resolution=True)
    data_vol = img.get_fdata()  # 3‑D float array

    # Global Z‑score → emphasises regions with strongest deviation
    src_map_z = stats.zscore(data_vol, axis=None)
    src_map_z = src_map_z.reshape(img.shape)

    return src_map_z, img.affine


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import nibabel as nib

    parser = argparse.ArgumentParser(description="Estimate iEEG sources via dSPM")
    parser.add_argument("--ieeg", required=True, help="Path to iEEG file (e.g. EDF)")
    parser.add_argument("--elecs", required=True, help="CSV with x,y,z[,name] cols")
    parser.add_argument("--mri", help="Subject T1.mgz (optional)")
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--out", default="ieeg_sources_z.nii.gz")
    args = parser.parse_args()

    loc_df = pd.read_csv(args.elecs)

    src_z, affine = estimate_sources_ieeg(
        raw=args.ieeg,
        electrode_locs=loc_df,
        mri_path=args.mri,
        time_window=(args.tmin, args.tmax) if args.tmin is not None else None,
    )

    nib.save(nib.Nifti1Image(src_z, affine), args.out)
    print(f"Saved Z‑scored source map → {args.out}")
