#!/usr/bin/env python3
"""
Furthest Point Sampling (FPS) for MACE-EXYZ datasets.

- Parses extended XYZ (EXYZ) with ASE.
- Builds per-structure features from:
  * global MACE descriptor fields in atoms.info
  * pooled stats over per-atom `mace_mp_descriptors` (overall + per species)
  * simple cell/composition scalars
- Standardizes (and optionally PCA-reduces) features (scikit-learn).
- Runs greedy k-center (furthest-point) sampling in feature space.
- Saves a CSV summary and an EXYZ with selected frames.

Dependencies: ase, numpy, pandas, scikit-learn
    pip install ase numpy pandas scikit-learn
"""

from __future__ import annotations
import argparse
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from ase.io import read, write
from ase.data import chemical_symbols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------
# Feature extraction utilities
# ----------------------------

GLOBAL_INFO_KEYS = [
    # present in your comment line; script will skip missing keys gracefully
    "mace_mp_descriptor",
    "mace_mp_Al_descriptor",
    "mace_mp_Ga_descriptor",
    "mace_mp_N_descriptor",
]

PER_ATOM_ARRAY_KEY = "mace_mp_descriptors"  # scalar per atom in your file


def safe_get(info: Dict[str, Any], key: str, default=np.nan) -> float:
    v = info.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def pooled_stats(values: np.ndarray, quantiles=(0.1, 0.5, 0.9)) -> List[float]:
    """Return [mean, std, q10, q50, q90] for a 1D array; NaNs if empty."""
    if values.size == 0:
        return [np.nan, np.nan] + [np.nan] * len(quantiles)
    m = float(np.mean(values))
    s = float(np.std(values))
    qs = np.quantile(values, quantiles).astype(float).tolist()
    return [m, s] + qs


def comp_fractions(symbols: List[str], species=("Al", "Ga", "N")) -> List[float]:
    total = len(symbols)
    if total == 0:
        return [np.nan] * len(species)
    out = []
    for sp in species:
        out.append(symbols.count(sp) / total)
    return out


def cell_features(atoms) -> List[float]:
    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    vol = atoms.get_volume()
    ca = c / a if a > 0 else np.nan
    ba = b / a if a > 0 else np.nan
    return [vol, a, b, c, ca, ba]


def per_species_pooled(values: np.ndarray, symbols: List[str], species=("Al", "Ga", "N")) -> List[float]:
    """For each species: mean, std, q10, q50, q90 (5 numbers per species)."""
    out: List[float] = []
    arr = np.asarray(values).reshape(-1)
    for sp in species:
        mask = np.array([s == sp for s in symbols], dtype=bool)
        out.extend(pooled_stats(arr[mask]))
    return out


def build_feature_vector(atoms) -> Tuple[List[float], List[str]]:
    """
    Construct a feature vector and the corresponding column names.
    Uses: global info keys, composition, cell scalars, pooled per-atom descriptor stats overall & per species.
    """
    syms = atoms.get_chemical_symbols()
    info = atoms.info if hasattr(atoms, "info") else {}
    arrs = atoms.arrays if hasattr(atoms, "arrays") else {}

    feats: List[float] = []
    cols: List[str] = []

    # 1) Global MACE fields
    for k in GLOBAL_INFO_KEYS:
        feats.append(safe_get(info, k, np.nan))
        cols.append(k)

    # 2) Composition fractions
    comp = comp_fractions(syms, species=("Al", "Ga", "N"))
    feats.extend(comp)
    cols.extend(["frac_Al", "frac_Ga", "frac_N"])

    # 3) Cell features
    cf = cell_features(atoms)
    feats.extend(cf)
    cols.extend(["vol", "a", "b", "c", "c_over_a", "b_over_a"])

    # 4) Per-atom descriptor pooled stats
    if PER_ATOM_ARRAY_KEY in arrs:
        # ensure 1D scalar per atom
        x = np.array(arrs[PER_ATOM_ARRAY_KEY]).reshape(-1)
        overall = pooled_stats(x)
        feats.extend(overall)
        cols.extend(["atomdesc_mean", "atomdesc_std", "atomdesc_q10", "atomdesc_q50", "atomdesc_q90"])

        # per-species pooled
        ps = per_species_pooled(x, syms, species=("Al", "Ga", "N"))
        feats.extend(ps)
        # make column names for each species block
        for sp in ("Al", "Ga", "N"):
            cols.extend([f"{sp}_mean", f"{sp}_std", f"{sp}_q10", f"{sp}_q50", f"{sp}_q90"])
    else:
        # if missing, fill NaNs to keep column alignment consistent
        feats.extend([np.nan] * 5)  # overall
        cols.extend(["atomdesc_mean", "atomdesc_std", "atomdesc_q10", "atomdesc_q50", "atomdesc_q90"])
        feats.extend([np.nan] * 15)  # 3 species * 5 stats
        for sp in ("Al", "Ga", "N"):
            cols.extend([f"{sp}_mean", f"{sp}_std", f"{sp}_q10", f"{sp}_q50", f"{sp}_q90"])

    return feats, cols


# ----------------------------
# FPS (greedy k-center)
# ----------------------------

def fps_indices(X: np.ndarray, k: int, start_index: int | None = None, rng: np.random.Generator | None = None) -> List[int]:
    """
    Furthest Point Sampling in Euclidean space without building full NxN distances.
    Returns indices of selected points (length k).
    """
    n = X.shape[0]
    if k >= n:
        return list(range(n))
    if rng is None:
        rng = np.random.default_rng(123)

    if start_index is None:
        # start from the point with the largest norm (often a good spreader)
        start_index = int(np.argmax(np.linalg.norm(X, axis=1)))
    sel = [start_index]

    # initialize min distances to +inf
    min_dist = np.full(n, np.inf, dtype=float)

    # iterate
    for _ in range(1, k):
        # compute squared distances to last selected only
        diff = X - X[sel[-1]]
        d2 = np.einsum("ij,ij->i", diff, diff)  # squared Euclidean
        # update nearest distance
        min_dist = np.minimum(min_dist, d2)
        # pick farthest by min_dist
        next_idx = int(np.argmax(min_dist))
        sel.append(next_idx)

    return sel


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="FPS selection from EXYZ using MACE descriptors + cell/composition features.")
    ap.add_argument("exyz", help="Input extended XYZ file (can contain multiple frames).")
    ap.add_argument("--k", type=int, required=True, help="Number of structures to select via FPS.")
    ap.add_argument("--out_xyz", default=None, help="Output EXYZ file with selected frames. Default: <stem>.fps.k<k>.xyz")
    ap.add_argument("--csv", default=None, help="Output CSV with features and selection flags. Default: <stem>.features.csv")
    ap.add_argument("--pca", type=int, default=0, help="Optional PCA dimension (0 = no PCA).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed (only used if start index not set).")
    ap.add_argument("--start_index", type=int, default=None, help="Optional index to start FPS from.")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.exyz))[0]
    out_xyz = args.out_xyz or f"{stem}.fps.k{args.k}.xyz"
    out_csv = args.csv or f"{stem}.features.csv"

    # Read all frames
    frames = read(args.exyz, ":")  # ASE reads all frames in EXYZ
    if len(frames) == 0:
        raise RuntimeError("No frames found in the input EXYZ.")

    # Build features
    feature_rows: List[List[float]] = []
    columns: List[str] | None = None
    for at in frames:
        feats, cols = build_feature_vector(at)
        feature_rows.append(feats)
        if columns is None:
            columns = cols
    X = np.array(feature_rows, dtype=float)
    df = pd.DataFrame(X, columns=columns)

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Optional PCA
    if args.pca and args.pca > 0:
        pca = PCA(n_components=args.pca, svd_solver="auto", random_state=args.seed)
        Xz = pca.fit_transform(Xs)
        df["__pca_var_explained__"] = np.nan  # placeholder for visibility
        explained = pca.explained_variance_ratio_.sum()
        print(f"[INFO] PCA reduced to {args.pca} dims; variance retained: {explained:.3f}")
    else:
        Xz = Xs

    # FPS
    rng = np.random.default_rng(args.seed)
    sel = fps_indices(Xz, k=args.k, start_index=args.start_index, rng=rng)

    # Save outputs
    # 1) EXYZ with selected frames
    write(out_xyz, [frames[i] for i in sel])
    # 2) CSV with features and selection flag
    df.insert(0, "index", np.arange(len(frames)))
    df["selected"] = False
    df.loc[sel, "selected"] = True
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Selected {len(sel)} / {len(frames)} structures.")
    print(f"       Wrote: {out_xyz}")
    print(f"       Wrote: {out_csv}")


if __name__ == "__main__":
    main()