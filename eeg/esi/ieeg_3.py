from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from mne.io import read_raw
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.transforms import Transform
from scipy import stats

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)

# ---------------------------------------------------------------------
# Dataclass configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the iEEG source estimation pipeline."""

    ieeg_path: Path
    electrodes_path: Path
    mri_path: Path | None = None
    subjects_dir: Path | None = None
    spacing_mm: float = 5.0
    inverse_method: str = "dSPM"
    snr: float = 3.0
    time_window: Tuple[float, float] | None = None
    loose: float = 0.2
    depth: float = 0.8
    out_file: Path = Path("ieeg_sources_z.nii.gz")
    verbose: bool = False
    #
    # --------------- NEW options ------------------------------------
    #
    coord_units: Literal["mm", "m"] = "mm"   # ← NEW
    keep_time: bool = False                  # ← NEW  (return full time course?)
    to_mni: bool = False                     # ← NEW  (morph to fsaverage/MNI)

# ---------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------

def load_raw(cfg: PipelineConfig) -> mne.io.BaseRaw:
    """Load iEEG data and ensure it's in memory."""
    LOG.info("Loading raw data → %s", cfg.ieeg_path)
    raw = read_raw(cfg.ieeg_path, preload=True, verbose=cfg.verbose)
    return raw


def load_electrodes(cfg: PipelineConfig, n_channels: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Read electrode coordinates CSV and return metres, names, and types."""
    LOG.info("Reading electrode coordinates → %s", cfg.electrodes_path)
    df = pd.read_csv(cfg.electrodes_path)

    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")

    scale = 0.001 if cfg.coord_units == "mm" else 1.0      # ← NEW
    coords = df[["x", "y", "z"]].to_numpy(float) * scale   # ← NEW scaling to metres

    if coords.shape[0] != n_channels:
        raise ValueError(
            f"Coordinate count {coords.shape[0]} ≠ n_channels {n_channels}"
        )

    names = (
        df["name"].astype(str).to_list()
        if "name" in df.columns
        else [f"ch{i}" for i in range(n_channels)]
    )

    types = (
        df["type"].str.lower().fillna("seeg").to_list()   # ← NEW; default seeg
        if "type" in df.columns
        else ["seeg"] * n_channels
    )
    return coords, names, types


def set_montage_and_types(raw: mne.io.BaseRaw,
                          coords: np.ndarray,
                          names: list[str],
                          types: list[str]) -> None:
    """Attach DigMontage and set channel types."""
    montage = mne.channels.make_dig_montage(
        ch_pos={n: c for n, c in zip(names, coords)}, coord_frame="mri"
    )
    raw.set_montage(montage, on_missing="ignore")

    # Ensure channel types are recognised by MNE forward routines
    type_map = {"seeg": "seeg", "ecog": "ecog"}
    ch_types = {name: type_map.get(t, "seeg") for name, t in zip(names, types)}
    raw.set_channel_types(ch_types, on_unit_change="ignore")  # safe in 1.9


def make_subject_id(cfg: PipelineConfig) -> str:
    """Infer subject ID (fallback: fsaverage)."""
    if cfg.mri_path is None:
        return "fsaverage"
    mri_path = Path(cfg.mri_path)
    for part in mri_path.parts:
        if part.startswith("sub-"):
            return part
    return mri_path.parent.name


def build_forward_model(
    cfg: PipelineConfig, raw: mne.io.BaseRaw, subject: str
) -> tuple[mne.Forward, list[mne.SourceSpaces]]:
    """Create volume source space, conductor model, and forward operator."""
    LOG.info("Setting up %s mm volumetric source space", cfg.spacing_mm)
    src = mne.setup_volume_source_space(
        subject=subject,
        pos=cfg.spacing_mm,
        subjects_dir=cfg.subjects_dir,
        add_interpolator=False,
        verbose=cfg.verbose,
    )

    # ---------------- conductor model ---------------------------------
    try:
        LOG.info("Attempting BEM model/solution (requires FreeSurfer surfaces)")
        bem = mne.make_bem_solution(
            mne.make_bem_model(
                subject=subject,
                subjects_dir=cfg.subjects_dir,
                conductivity=(0.3,),
                verbose=cfg.verbose,
            )
        )
    except FileNotFoundError:
        LOG.warning("BEM surfaces not found → falling back to spherical model")
        bem = mne.make_sphere_model("auto", "auto", raw.info)   # ← NEW

    trans = Transform("mri", "head")  # identity; electrodes in MRI coords

    LOG.info("Computing forward solution (seeg+ecog enabled)")
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False, eeg=False,
        seeg=True, ecog=True,    # ← NEW flags (supported ≥ 1.6, still valid in 1.9)
        verbose=cfg.verbose,
    )
    return fwd, src


def create_evoked_or_epochs(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> mne.Evoked | mne.Epochs:
    """Return either an Evoked (mean) or raw-like Epochs object for time‑resolved output."""
    if cfg.time_window is not None:
        tmin, tmax = cfg.time_window
        LOG.info("Cropping raw data to %.3f–%.3f s", tmin, tmax)
        raw = raw.copy().crop(tmin, tmax)

    if cfg.keep_time:          # ← NEW branch
        # Single epoch holding the continuous data
        events = np.array([[0, 0, 1]])
        epochs = mne.EpochsArray(
            raw.get_data()[None, :, :], raw.info, events, tmin=0.0, verbose=False
        )
        return epochs
    else:
        mean_data = raw.get_data().mean(axis=1, keepdims=True)
        return mne.EvokedArray(mean_data, raw.info, tmin=0.0, comment="iEEG‑mean")


def estimate_sources(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full pipeline and return (z‑map, affine)."""
    raw = load_raw(cfg)
    coords, names, types = load_electrodes(cfg, n_channels=len(raw.ch_names))
    set_montage_and_types(raw, coords, names, types)

    subject = make_subject_id(cfg)
    fwd, src = build_forward_model(cfg, raw, subject)

    data_obj = create_evoked_or_epochs(raw, cfg)

    LOG.info("Creating diagonal noise covariance")
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=cfg.verbose)

    lambda2 = 1.0 / cfg.snr**2
    inverse_operator = make_inverse_operator(
        data_obj.info,
        fwd,
        noise_cov,
        loose=cfg.loose,
        depth=cfg.depth,
        verbose=cfg.verbose,
    )

    LOG.info("Applying inverse solution (%s, λ²=%.3g)", cfg.inverse_method, lambda2)
    stc = apply_inverse(
        data_obj,
        inverse_operator,
        lambda2=lambda2,
        method=cfg.inverse_method,
        pick_ori=None,
        verbose=cfg.verbose,
    )

    # ---------------- Optional morph to MNI ---------------------------
    if cfg.to_mni and subject != "fsaverage":
        LOG.info("Morphing volumetric STC to fsaverage (MNI)")
        morph = mne.compute_source_morph(
            stc, subject_from=subject, subject_to="fsaverage",
            subjects_dir=cfg.subjects_dir, verbose=cfg.verbose
        )
        stc = morph.apply(stc)

    LOG.info("Converting to volume & Z‑scoring")
    img = stc.as_volume(src, mri_resolution=True)
    data = img.get_fdata().astype(np.float32)

    # Z‑score across all voxels & time
    zmap = stats.zscore(data.reshape(-1), axis=None, nan_policy="omit").reshape(data.shape)
    return zmap, img.affine


def save_nifti(zmap: np.ndarray, affine: np.ndarray, out_file: Path) -> None:
    LOG.info("Saving Z‑scored source map → %s", out_file)
    nib.save(nib.Nifti1Image(zmap, affine), str(out_file))


# ------------------- placeholder for future stats --------------------
def cluster_zmap(zmap: np.ndarray, threshold: float = 3.1) -> None:  # ← NEW
    """Stub for cluster‑based thresholding (future work)."""
    pass

# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        description="Estimate iEEG volumetric sources (functional pipeline)"
    )
    p.add_argument("--ieeg", required=True, type=Path, help="Path to iEEG file")
    p.add_argument("--elecs", required=True, type=Path, help="Electrode CSV (x,y,z[,name,type])")
    p.add_argument("--mri", type=Path, help="T1.mgz (optional) — defaults to fsaverage")
    p.add_argument("--tmin", type=float, help="Start time (s) to average")
    p.add_argument("--tmax", type=float, help="End time (s) to average")
    p.add_argument("--spacing", type=float, default=5.0, help="Source grid (mm)")
    p.add_argument("--snr", type=float, default=3.0, help="SNR (λ²=1/snr²)")
    p.add_argument("--method", choices=["MNE", "dSPM", "sLORETA"], default="dSPM")
    p.add_argument("--subjects-dir", type=Path, dest="subjects_dir", help="$SUBJECTS_DIR override")
    p.add_argument("--out", type=Path, default=Path("ieeg_sources_z.nii.gz"), help="Output NIfTI")
    #
    # ---------------- NEW CLI flags -----------------------------------
    #
    p.add_argument("--units", choices=["mm", "m"], default="mm",
                   help="Units of x,y,z in electrode CSV (default: mm)")      # ← NEW
    p.add_argument("--keep-time", action="store_true",
                   help="Keep full time course instead of time‑average")      # ← NEW
    p.add_argument("--to-mni", action="store_true",
                   help="Morph result to fsaverage (MNI) before saving")      # ← NEW
    p.add_argument("--verbose", action="store_true",
                   help="Verbose MNE output")                                 # ← NEW

    args = p.parse_args()
    cfg = PipelineConfig(
        ieeg_path=args.ieeg,
        electrodes_path=args.elecs,
        mri_path=args.mri,
        subjects_dir=args.subjects_dir,
        spacing_mm=args.spacing,
        inverse_method=args.method,
        snr=args.snr,
        time_window=(args.tmin, args.tmax) if args.tmin is not None else None,
        out_file=args.out,
        #
        coord_units=args.units,      # ← NEW
        keep_time=args.keep_time,    # ← NEW
        to_mni=args.to_mni,          # ← NEW
        verbose=args.verbose,        # ← NEW
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    zmap, affine = estimate_sources(cfg)
    save_nifti(zmap, affine, cfg.out_file)
    LOG.info("Done!")


if __name__ == "__main__":
    main()
