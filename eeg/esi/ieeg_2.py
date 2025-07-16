from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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


# ---------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------

def load_raw(cfg: PipelineConfig) -> mne.io.BaseRaw:
    """Load iEEG data and ensure it's in memory."""
    LOG.info("Loading raw data → %s", cfg.ieeg_path)
    raw = read_raw(cfg.ieeg_path, preload=True, verbose=cfg.verbose)
    return raw


def load_electrodes(cfg: PipelineConfig, n_channels: int) -> tuple[np.ndarray, list[str]]:
    """Read electrode coordinates CSV → (n, 3) in metres + names."""
    LOG.info("Reading electrode coordinates → %s", cfg.electrodes_path)
    df = pd.read_csv(cfg.electrodes_path)
    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")
    coords = df[["x", "y", "z"]].to_numpy(float)
    if coords.shape[0] != n_channels:
        raise ValueError(
            f"Coordinate count {coords.shape[0]} ≠ n_channels {n_channels}"
        )
    names = (
        df["name"].astype(str).to_list()
        if "name" in df.columns
        else [f"ch{i}" for i in range(n_channels)]
    )
    return coords, names


def set_montage(raw: mne.io.BaseRaw, coords: np.ndarray, names: list[str]) -> None:
    """Attach DigMontage in MRI coordinates (identity trans)."""
    montage = mne.channels.make_dig_montage(
        ch_pos={n: c for n, c in zip(names, coords)}, coord_frame="mri"
    )
    raw.set_montage(montage, on_missing="ignore")


def make_subject_id(cfg: PipelineConfig) -> str:
    """Infer subject ID (fallback: fsaverage)."""
    if cfg.mri_path is None:
        return "fsaverage"
    mri_path = Path(cfg.mri_path)
    # e.g. derivatives/freesurfer/sub‑01/mri/T1.mgz  →  sub‑01
    for part in mri_path.parts:
        if part.startswith("sub-"):
            return part
    return mri_path.parent.name  # last dir name


def build_forward_model(
    cfg: PipelineConfig, raw: mne.io.BaseRaw, subject: str
) -> tuple[mne.Forward, list[mne.SourceSpaces]]:
    """Create volume source space, BEM and forward operator."""
    LOG.info("Setting up %s mm volumetric source space", cfg.spacing_mm)
    src = mne.setup_volume_source_space(
        subject=subject,
        pos=cfg.spacing_mm,
        subjects_dir=cfg.subjects_dir,
        add_interpolator=False,
        verbose=cfg.verbose,
    )

    LOG.info("Building BEM model/solution")
    bem = mne.make_bem_solution(
        mne.make_bem_model(
            subject=subject,
            subjects_dir=cfg.subjects_dir,
            conductivity=(0.3,),
            verbose=cfg.verbose,
        )
    )

    trans = Transform("mri", "head")  # identity: electrodes already in MRI coords

    LOG.info("Computing forward solution")
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=False,
        verbose=cfg.verbose,
    )
    return fwd, src


def create_evoked(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> mne.Evoked:
    """Average data within *time_window* (or full recording)."""
    if cfg.time_window is not None:
        tmin, tmax = cfg.time_window
        LOG.info("Cropping raw data to %.3f–%.3f s", tmin, tmax)
        raw = raw.copy().crop(tmin, tmax)
    mean_data = raw.get_data().mean(axis=1, keepdims=True)
    evoked = mne.EvokedArray(mean_data, raw.info, tmin=0.0, comment="iEEG‑mean")
    return evoked


def estimate_sources(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full pipeline and return (z‑map, affine)."""
    raw = load_raw(cfg)
    coords, names = load_electrodes(cfg, n_channels=len(raw.ch_names))
    set_montage(raw, coords, names)

    subject = make_subject_id(cfg)
    fwd, src = build_forward_model(cfg, raw, subject)

    evoked = create_evoked(raw, cfg)

    LOG.info("Creating diagonal noise covariance")
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=cfg.verbose)

    lambda2 = 1.0 / cfg.snr**2
    inverse_operator = make_inverse_operator(
        evoked.info,
        fwd,
        noise_cov,
        loose=cfg.loose,
        depth=cfg.depth,
        verbose=cfg.verbose,
    )

    LOG.info(
        "Applying inverse solution (%s, λ²=%.3g)" % (cfg.inverse_method, lambda2)
    )
    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=cfg.inverse_method,
        pick_ori=None,
        verbose=cfg.verbose,
    )

    LOG.info("Converting to volume & Z‑scoring")
    img = stc.as_volume(src, mri_resolution=True)
    data = img.get_fdata().astype(np.float32)
    zmap = stats.zscore(data, axis=None, nan_policy="omit").reshape(img.shape)
    return zmap, img.affine


def save_nifti(zmap: np.ndarray, affine: np.ndarray, out_file: Path) -> None:
    LOG.info("Saving Z‑scored source map → %s", out_file)
    nib.save(nib.Nifti1Image(zmap, affine), str(out_file))


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        description="Estimate iEEG volumetric sources (functional pipeline)"
    )
    p.add_argument("--ieeg", required=True, type=Path, help="Path to iEEG file")
    p.add_argument("--elecs", required=True, type=Path, help="Electrode CSV (x,y,z[,name])")
    p.add_argument("--mri", type=Path, help="T1.mgz (optional) — defaults to fsaverage")
    p.add_argument("--tmin", type=float, help="Start time (s) to average")
    p.add_argument("--tmax", type=float, help="End time (s) to average")
    p.add_argument("--spacing", type=float, default=5.0, help="Source grid (mm)")
    p.add_argument("--snr", type=float, default=3.0, help="SNR (λ²=1/snr²)")
    p.add_argument("--method", choices=["MNE", "dSPM", "sLORETA"], default="dSPM")
    p.add_argument("--subjects-dir", type=Path, dest="subjects_dir", help="$SUBJECTS_DIR override")
    p.add_argument("--out", type=Path, default=Path("ieeg_sources_z.nii.gz"), help="Output NIfTI")

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
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    zmap, affine = estimate_sources(cfg)
    save_nifti(zmap, affine, cfg.out_file)
    LOG.info("Done!")


if __name__ == "__main__":
    main()