import os
import mne
import pandas as pd


def load_electrodes(path, coord_units, n_channels):
    df = pd.read_csv(path)
    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")
    scale = 0.001 if coord_units == "mm" else 1.0  
    coords = df[["x", "y", "z"]].to_numpy(float) * scale
    if coords.shape[0] != n_channels:
        raise ValueError(f"Coordinate count {coords.shape[0]} ≠ n_channels {n_channels}")
    names = (
        df["name"].astype(str).to_list()
        if "name" in df.columns
        else [f"ch{i}" for i in range(n_channels)]
    )
    types = (
        df["type"].str.lower().fillna("seeg").to_list()
        if "type" in df.columns
        else ["seeg"] * n_channels
    )
    return coords, names, types