import os
import sys
import shutil
import warnings
import subprocess
import numpy as np
import nibabel as nib
from scipy import stats
from pathlib import Path


def _which(exe):
    return shutil.which(exe) is not None

def _load(path):
    img = nib.load(str(path))
    return img, img.get_fdata(dtype=np.float32)

def _save_like(ref_img, data, out_path, dtype=np.float32):
    out_img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))
    return str(out_path)

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{r.stderr}")
    return r

def _find_first(root, names):
    for n in names:
        p = Path(root) / n
        if p.exists():
            return p
    return None

def _download_iit(target_dir):
    """
    Try to fetch IIT v5 maps needed for z-scoring.
    If your environment needs NITRC login, you may have to log in in a browser
    and place these files manually into target_dir.
    """
    target_dir = Path(target_dir)
    _ensure_dir(target_dir)
    files = {
        "IITmean_FA.nii.gz":      "https://www.nitrc.org/frs/download.php/11271/IITmean_FA.nii.gz",
        "IITmean_fastd.nii.gz":   "https://www.nitrc.org/frs/download.php/11272/IITmean_fastd.nii.gz",  # FA SD
        "IITmean_tr.nii.gz":      "https://www.nitrc.org/frs/download.php/11278/IITmean_tr.nii.gz",     # trace mean
        "IITmean_trstd.nii.gz":   "https://www.nitrc.org/frs/download.php/11279/IITmean_trstd.nii.gz",  # trace SD
    }
    try:
        import requests
    except Exception:
        print("[soz] 'requests' not available; skipping auto-download.")
        return {k: target_dir / k for k in files}

    out_paths = {}
    for fname, url in files.items():
        out = target_dir / fname
        out_paths[fname] = out
        if out.exists():
            continue
        try:
            print(f"[soz] Downloading {fname} from {url} ...")
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                out.write_bytes(resp.content)
                print(f"[soz]   saved → {out}")
            else:
                print(f"[soz]   HTTP {resp.status_code} for {url}. Please download manually to {out}")
        except Exception as e:
            print(f"[soz]   download failed: {e}. Please download manually to {out}")
    return out_paths


def build_soz_zmap(
    subject_dir,
    output_dir=None,
    patient_age=None,
    ref_source="IIT_v5",          # or "local"
    norm_dir="normative_iit",
    control_fa_list=None,         # optional: list of FA paths (already in the same template space) for local controls
    control_md_list=None,         # optional: MD paths (same length & space as FA list)
    fa_weight=1.0,
    md_weight=1.0,
    clip_negatives=True,
    keep_intermediates=False
):
    
    subj_root = Path(subject_dir)
    analyzed = subj_root if (subj_root.name == "analyzed_fsl") else (subj_root / "analyzed_fsl")
    if not analyzed.exists():
        raise FileNotFoundError(f"Cannot find analyzed_fsl at {analyzed}")

    # Required inputs
    fa_p = _find_first(analyzed, ["dti_FA.nii.gz"])
    if fa_p is None:
        raise FileNotFoundError("Missing dti_FA.nii.gz in analyzed_fsl")
    md_p = _find_first(analyzed, ["dti_MD.nii.gz"])
    l1 = _find_first(analyzed, ["dti_L1.nii.gz"])
    l2 = _find_first(analyzed, ["dti_L2.nii.gz"])
    l3 = _find_first(analyzed, ["dti_L3.nii.gz"])
    mask_p = _find_first(analyzed, ["brain_mask.nii.gz", "nodif_brain_posteddy_mask.nii.gz"])
    if mask_p is None:
        raise FileNotFoundError("Missing brain mask in analyzed_fsl")

    # If MD missing, derive from L1+L2+L3
    if md_p is None:
        if not (l1 and l2 and l3):
            raise FileNotFoundError("Missing dti_MD.nii.gz and cannot derive from L1/L2/L3")
        print("[soz] Deriving MD from eigenvalues...")
        l1_i, L1 = _load(l1)
        L2 = nib.load(str(l2)).get_fdata(dtype=np.float32)
        L3 = nib.load(str(l3)).get_fdata(dtype=np.float32)
        MD = (L1 + L2 + L3) / 3.0
        md_p = analyzed / "dti_MD.nii.gz"
        _save_like(l1_i, MD, md_p)

    # Set output folder
    out_dir = Path(output_dir) if output_dir else (analyzed / "soz_maps")
    _ensure_dir(out_dir)

    # Warn if age far from IIT adult range
    if patient_age is not None and not (20 <= int(patient_age) <= 40):
        warnings.warn(
            f"Patient age {patient_age}y is outside IIT adult range (20–40). "
            "Proceeding, but consider building a matched local control set."
        )

    # Reference maps
    if ref_source == "IIT_v5":
        # Make sure we have the IIT maps
        paths = _download_iit(norm_dir)
        iit_fa  = Path(paths["IITmean_FA.nii.gz"])
        iit_fasd= Path(paths["IITmean_fastd.nii.gz"])
        iit_tr  = Path(paths["IITmean_tr.nii.gz"])
        iit_trsd= Path(paths["IITmean_trstd.nii.gz"])
        for p in [iit_fa, iit_fasd, iit_tr, iit_trsd]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Reference file missing: {p}. If auto-download failed, "
                    f"download from NITRC and place it here: {Path(norm_dir).resolve()}"
                )
        ref_space_img, _ = _load(iit_fa)
        # MD mean/SD from trace (tr = 3*MD)
        iit_tr_img, TR = _load(iit_tr)
        _, TRSD = _load(iit_trsd)
        MD_mean = TR / 3.0
        MD_sd   = TRSD / 3.0
        _, FA_mean = _load(iit_fa)
        _, FA_sd   = _load(iit_fasd)  # "fastd" is FA SD
    elif ref_source == "local":
        if not (control_fa_list and control_md_list and len(control_fa_list) == len(control_md_list) > 2):
            raise ValueError("Provide >=3 control FA/MD maps (same space) when ref_source='local'.")
        # Assume controls are already in the same template space; check dimensions
        first_fa_img, first_fa = _load(control_fa_list[0])
        ref_space_img = first_fa_img
        FA_stack = np.stack([nib.load(p).get_fdata(dtype=np.float32) for p in control_fa_list], axis=0)
        MD_stack = np.stack([nib.load(p).get_fdata(dtype=np.float32) for p in control_md_list], axis=0)
        FA_mean, FA_sd = FA_stack.mean(0), FA_stack.std(0, ddof=1)
        MD_mean, MD_sd = MD_stack.mean(0), MD_stack.std(0, ddof=1)
    else:
        raise ValueError("ref_source must be 'IIT_v5' or 'local'")

    # Register subject FA -> reference (IIT or local template)
    if not _which("flirt"):
        raise EnvironmentError("FSL FLIRT not found in PATH. Please install FSL and ensure 'flirt' is available.")
    work = out_dir / "work"
    _ensure_dir(work)

    fa_in = fa_p
    md_in = md_p
    mask_in = mask_p

    fa_reg = work / "fa_in_ref.nii.gz"
    md_reg = work / "md_in_ref.nii.gz"
    msk_reg = work / "mask_in_ref.nii.gz"
    mat = work / "fa2ref.mat"

    # Estimate transform using FA
    _run([
        "flirt", "-in", str(fa_in), "-ref", str(ref_space_img.get_filename()),
        "-omat", str(mat), "-dof", "12", "-cost", "corratio", "-out", str(fa_reg)
    ])
    # Apply to MD and mask
    _run([
        "flirt", "-in", str(md_in), "-ref", str(ref_space_img.get_filename()),
        "-applyxfm", "-init", str(mat), "-out", str(md_reg)
    ])
    _run([
        "flirt", "-in", str(mask_in), "-ref", str(ref_space_img.get_filename()),
        "-applyxfm", "-init", str(mat), "-interp", "nearestneighbour", "-out", str(msk_reg)
    ])

    # Compute z maps within mask
    _, FA_subj = _load(fa_reg)
    _, MD_subj = _load(md_reg)
    _, MASK = _load(msk_reg)
    MASK = (MASK > 0.5).astype(np.float32)

    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        z_FA_low = (FA_mean - FA_subj) / (FA_sd + eps)   # low FA abnormal
        z_MD_high = (MD_subj - MD_mean) / (MD_sd + eps)  # high MD abnormal
        if clip_negatives:
            z_FA_low[z_FA_low < 0] = 0
            z_MD_high[z_MD_high < 0] = 0
        # Combine (weighted additive; both emphasize pathology-consistent directions)
        soz = fa_weight * z_FA_low + md_weight * z_MD_high

    # Mask outside brain
    z_FA_low *= MASK
    z_MD_high *= MASK
    soz *= MASK

    # Save outputs
    out_paths = {}
    out_paths["z_FA_low"] = _save_like(ref_space_img, z_FA_low, out_dir / "z_FA_low.nii.gz")
    out_paths["z_MD_high"] = _save_like(ref_space_img, z_MD_high, out_dir / "z_MD_high.nii.gz")
    out_paths["soz_zmap"]  = _save_like(ref_space_img, soz, out_dir / "soz_zmap.nii.gz")

    # Optional: ROI summary using Brainnetome atlas resampled to reference space
    atlas_dwi = analyzed / "atlas_in_dwi.nii.gz"
    if atlas_dwi.exists():
        atlas_ref = work / "atlas_in_ref.nii.gz"
        _run([
            "flirt", "-in", str(atlas_dwi), "-ref", str(ref_space_img.get_filename()),
            "-applyxfm", "-init", str(mat), "-interp", "nearestneighbour", "-out", str(atlas_ref)
        ])
        atlas_img, ATLAS = _load(atlas_ref)
        labels = np.unique(ATLAS.astype(np.int32))
        labels = labels[labels > 0]
        rows = ["roi,label_vox,mean_FA,mean_MD,mean_z_FA_low,mean_z_MD_high,mean_z_SOZ"]
        for lab in labels:
            m = (ATLAS == lab)
            if m.sum() == 0: 
                continue
            mean_fa = FA_subj[m].mean()
            mean_md = MD_subj[m].mean()
            mean_zfa = z_FA_low[m].mean()
            mean_zmd = z_MD_high[m].mean()
            mean_soz = soz[m].mean()
            rows.append(f"{lab},{int(m.sum())},{mean_fa:.6f},{mean_md:.6f},{mean_zfa:.6f},{mean_zmd:.6f},{mean_soz:.6f}")
        roi_csv = out_dir / "roi_stats.csv"
        roi_csv.write_text("\n".join(rows))
        out_paths["roi_stats"] = str(roi_csv)

    if not keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)

    return out_paths


if __name__ == "__main__":
    subject_dir = "/media/local1/Datasets/epilepsy_dti_fallahi/subj_01"
    norm_dir="datasets/normative_iit"
    print(build_soz_zmap(subject_dir=subject_dir, norm_dir=norm_dir))