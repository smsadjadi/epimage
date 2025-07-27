import argparse
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import mne
from nilearn import plotting, surface


###############################################################################
# I/O utilities
###############################################################################

def load_ieeg(raw_path: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load iEEG file using MNE‑Python (supports BIDS/EDF/FIF/BrainVision)."""
    raw = mne.io.read_raw(raw_path, preload=preload)
    if not any(ch['kind'] == mne.io.constants.FIFF.FIFFV_SEEG for ch in raw.info['chs']):
        print('⚠ No SEEG/ECoG channel type found – continuing but be sure montage is correct.')
    return raw


def load_montage(coords_path: Path) -> np.ndarray:
    """Load M ×\xa03 array of contact coordinates (in MRI RAS mm)."""
    ext = coords_path.suffix.lower()
    if ext == '.tsv':
        coords = np.loadtxt(coords_path, delimiter='\t')
    elif ext in ('.csv', '.txt'):
        coords = np.loadtxt(coords_path, delimiter=',')
    elif ext == '.json':
        coords = np.array(json.loads(coords_path.read_text()))
    else:
        raise ValueError(f'Unsupported coordinate format: {ext}')
    assert coords.shape[1] == 3, 'Coordinates must be in N×3 format (x, y, z, mm)'
    return coords


###############################################################################
# Signal‑level metrics
###############################################################################

def bandpass_power(raw: mne.io.BaseRaw, fmin, fmax, tmin, tmax) -> np.ndarray:
    """Compute band‑limited power per channel in [tmin, tmax] (sec)."""
    raw_filt = raw.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
    data, times = raw_filt.get_data(start=int(tmin*raw.info['sfreq']),
                                    stop=int(tmax*raw.info['sfreq']),
                                    return_times=True)
    analytic = mne.time_frequency.hilbert(raw_filt, envelope=True).get_data()
    power = analytic.mean(axis=1)  # mean envelope
    return power


###############################################################################
# Volume interpolation
###############################################################################

def interpolate_volume(coords_mm: np.ndarray,
                       values: np.ndarray,
                       mri_img: nib.Nifti1Image,
                       sigma: float = 5.0) -> nib.Nifti1Image:
    """Interpolate electrode values into MRI volume using Gaussian kernel.
    
    Parameters
    ----------
    coords_mm : (N, 3) array in MRI RAS mm
    values    : (N,) array of per‑electrode metrics
    mri_img   : nibabel NIfTI image of subject T1
    sigma     : spatial smoothing kernel (mm)
    """
    # Convert coords from RAS mm to voxel indices
    ras2vox = np.linalg.inv(mri_img.affine)
    vox_coords = nib.affines.apply_affine(ras2vox, coords_mm)
    vox_coords = np.asarray(vox_coords)
    
    vol = np.zeros(mri_img.shape, dtype=np.float32)
    weight = np.zeros_like(vol)
    
    print('→ Interpolating {} contacts …'.format(len(values)))
    for coord, val in zip(vox_coords, values):
        x, y, z = np.round(coord).astype(int)
        if not ((0 <= x < vol.shape[0]) and (0 <= y < vol.shape[1]) and (0 <= z < vol.shape[2])):
            print(f'⚠ Contact outside MRI bounds at voxel {coord}')
            continue
        vol[x, y, z] = val
        weight[x, y, z] = 1.0
    
    # Smooth with Gaussian kernel to propagate values
    vox_sigma = sigma / np.mean(np.diag(mri_img.header.get_zooms()))  # mm → vox
    vol_smooth = gaussian_filter(vol, vox_sigma, mode='constant')
    weight_smooth = gaussian_filter(weight, vox_sigma, mode='constant')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_smooth /= np.maximum(weight_smooth, 1e-12)
    vol_smooth[np.isnan(vol_smooth)] = 0
    return nib.Nifti1Image(vol_smooth, mri_img.affine, mri_img.header)


###############################################################################
# Command‑line interface
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='iEEG source mapping to MRI.')
    parser.add_argument('--ieeg', required=True, type=Path, help='Path to iEEG recording (EDF/BIDS/FIF/…)')
    parser.add_argument('--coords', required=True, type=Path, help='Electrode coordinates file (TSV/CSV/JSON)')
    parser.add_argument('--mri', required=True, type=Path, help='Subject T1w MRI NIfTI')
    parser.add_argument('--tmin', type=float, default=0.0, help='Window start (sec)')
    parser.add_argument('--tmax', type=float, default=10.0, help='Window end (sec)')
    parser.add_argument('--band', nargs=2, type=float, default=[70, 150], help='Frequency band (Hz) e.g. 70 150')
    parser.add_argument('--sigma', type=float, default=5.0, help='Gaussian kernel size (mm)')
    parser.add_argument('--out', type=Path, default=Path('ieeg_activation.nii.gz'), help='Output NIfTI')
    args = parser.parse_args()
    
    raw = load_ieeg(args.ieeg)
    coords = load_montage(args.coords)
    mri_img = nib.load(args.mri)
    
    values = bandpass_power(raw, *args.band, args.tmin, args.tmax)
    
    act_img = interpolate_volume(coords, values, mri_img, args.sigma)
    nib.save(act_img, args.out)
    print(f'✔ Activation map saved to {args.out}')
    
    # Quick visualisation
    plotting.plot_stat_map(act_img, bg_img=mri_img,
                           title='iEEG activation ({}–{} Hz)'.format(*args.band))
    plotting.show()
    
    # (Optional) surface projection
    try:
        print('→ Projecting to cortical surface …')
        surf_lh = surface.vol_to_surf(act_img, 'fsaverage:lh', interpolation='nearest')
        plotting.plot_surf_stat_map('fsaverage:lh', surf_lh, hemi='left',
                                    colorbar=True, title='Activation – left hemi')
        plotting.show()
    except Exception as exc:
        print('Surface projection skipped:', exc)


if __name__ == '__main__':
    main()
