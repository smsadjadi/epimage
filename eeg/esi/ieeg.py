from __future__ import annotations

import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Literal

import mne
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from mne.io import read_raw
from mne.transforms import Transform
from mne.minimum_norm import apply_inverse, make_inverse_operator

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
    coord_units: Literal["mm", "m"] = "mm"
    keep_time: bool = False

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

    scale = 0.001 if cfg.coord_units == "mm" else 1.0  
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
        ch_pos={n: c for n, c in zip(names, coords)}, coord_frame="head"
    )
    raw.set_montage(montage, on_missing="ignore")

    # Ensure channel types are recognised by MNE forward routines
    type_map = {"seeg": "seeg", "ecog": "ecog"}
    ch_types = {name: type_map.get(t, "seeg") for name, t in zip(names, types)}
    raw.set_channel_types(ch_types, on_unit_change="ignore")  # safe in 1.9


def make_subject_id(cfg: PipelineConfig) -> str:
    if cfg.mri_path is None:
        raise RuntimeError("Individual T1.mgz is required; please supply --mri <path>")
    mri_path = Path(cfg.mri_path)
    for part in mri_path.parts:
        if part.startswith("sub-"):
            return part
    return mri_path.parent.name


def build_forward_model(
    cfg: PipelineConfig, raw: mne.io.BaseRaw, subject: str
) -> tuple[mne.Forward, list[mne.SourceSpaces]]:
    """Create volume source space, conductor model, and forward operator."""
    if cfg.mri_path is not None:
        fs_mri_dir = Path(cfg.subjects_dir) / subject / "mri"
        fs_mri_dir.mkdir(parents=True, exist_ok=True)
        tgt = fs_mri_dir / "T1.mgz"

        if cfg.mri_path.suffix not in (".mgz",) or not tgt.is_file():
            img = nib.load(str(cfg.mri_path))
            nib.save(img, str(tgt))
            LOG.info(f"Converted {cfg.mri_path.name} → {tgt}")
        mri_for_mne = str(tgt)
        print(mri_for_mne)
    else:
        mri_for_mne = None

    LOG.info("Setting up %s mm volumetric source space", cfg.spacing_mm)
    src = mne.setup_volume_source_space(
        subject=subject,
        pos=cfg.spacing_mm,
        subjects_dir=cfg.subjects_dir,
        add_interpolator=True,
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

    trans = Transform("head", "mri")  # identity; electrodes in MRI coords

    LOG.info("Computing forward solution (seeg+ecog enabled)")
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        verbose=cfg.verbose,
    )
    return fwd, src


def create_evoked_or_epochs(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> mne.Evoked | mne.Epochs:
    """Return either an Evoked (mean) or raw-like Epochs object for time‑resolved output."""
    if cfg.time_window is not None:
        tmin, tmax = cfg.time_window
        LOG.info("Cropping raw data to %.3f–%.3f s", tmin, tmax)
        raw = raw.copy().crop(tmin, tmax)

    if cfg.keep_time:      
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

    _df = pd.read_csv(cfg.electrodes_path)
    _n_contacts = len(_df)
    coords, names, types = load_electrodes(cfg, n_channels=_n_contacts)
    raw.pick_channels(names, ordered=True)



def estimate_sources(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full pipeline and return (z-map, affine)."""
    raw = load_raw(cfg)

    _df = pd.read_csv(cfg.electrodes_path)
    _n_contacts = len(_df)
    coords, names, types = load_electrodes(cfg, n_channels=_n_contacts)
    raw.pick_channels(names, ordered=True)

    set_montage_and_types(raw, coords, names, types)
    raw.set_channel_types({ch: "eeg" for ch in names})
    raw.pick_channels(names, ordered=True)

    raw.set_eeg_reference('average', projection=True)

    subject = make_subject_id(cfg)
    fwd, src = build_forward_model(cfg, raw, subject)

    data_obj = create_evoked_or_epochs(raw, cfg)
    data_obj.apply_proj()

    LOG.info("Creating diagonal noise covariance")
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=cfg.verbose)

    lambda2 = 1.0 / cfg.snr**2

    # For volumetric source spaces, loose must be 1 or "auto"
    # Detect volume grids by ss['type'] == 'vol'
    is_volume = any(ss.get('type', None) == 'vol' for ss in src)
    loose_param = "auto" if is_volume else cfg.loose

    inverse_operator = make_inverse_operator(
        data_obj.info,
        fwd,
        noise_cov,
        loose=loose_param,
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
    p.add_argument("--mri", required=True, type=Path, help="Path to subject‑specific T1.mgz")
    p.add_argument("--tmin", type=float, help="Start time (s) to average")
    p.add_argument("--tmax", type=float, help="End time (s) to average")
    p.add_argument("--spacing", type=float, default=5.0, help="Source grid (mm)")
    p.add_argument("--snr", type=float, default=3.0, help="SNR (λ²=1/snr²)")
    p.add_argument("--method", choices=["MNE", "dSPM", "sLORETA"], default="dSPM")
    p.add_argument("--subjects-dir", type=Path, dest="subjects_dir", help="$SUBJECTS_DIR override")
    p.add_argument("--out", type=Path, default=Path("ieeg_sources_z.nii.gz"), help="Output NIfTI")
    p.add_argument("--units", choices=["mm", "m"], default="mm",help="Units of x,y,z in electrode CSV (default: mm)")  
    p.add_argument("--keep-time", action="store_true", help="Keep full time course instead of time‑average")  
    p.add_argument("--verbose", action="store_true", help="Verbose MNE output")                             

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
        coord_units=args.units,  
        keep_time=args.keep_time,
        verbose=args.verbose,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    zmap, affine = estimate_sources(cfg)
    save_nifti(zmap, affine, cfg.out_file)
    LOG.info("Done!")


if __name__ == "__main__":
    main()
