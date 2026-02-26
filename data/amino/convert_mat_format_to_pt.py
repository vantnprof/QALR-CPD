#!/usr/bin/env python3
"""
Convert amino.mat to a .pt payload compatible with synthetic datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import scipy.io as sio
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("scipy is required to load .mat files. Please install scipy and retry.") from exc


def _dtype_from_str(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{raw}'.")
    return mapping[key]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _load_dims(mat: Dict) -> Tuple[int, ...]:
    dimx = mat.get("DimX", None)
    if dimx is None:
        return ()
    dims = tuple(int(x) for x in np.array(dimx).ravel().tolist())
    return dims


def _reshape_X(X: np.ndarray, dims: Tuple[int, ...]) -> np.ndarray:
    if not dims:
        return X
    if int(np.prod(dims)) != int(X.size):
        raise ValueError(f"DimX product {np.prod(dims)} does not match X size {X.size}.")
    if X.ndim == len(dims) and tuple(X.shape) == dims:
        return X
    if X.ndim == 2:
        # MATLAB column-major layout: use Fortran order to recover original tensor.
        return np.reshape(X, dims, order="F")
    raise ValueError(f"Unexpected X shape {X.shape} for dims {dims}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert amino.mat to a .pt dataset payload.")
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=Path("data/Amino acid fluorescence/amino.mat"),
        help="Path to amino.mat.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/Amino acid fluorescence/amino.pt"),
        help="Output .pt path.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Output tensor dtype.")
    parser.add_argument("--seed", type=int, default=0, help="Seed stored in payload metadata.")
    args = parser.parse_args()

    mat = sio.loadmat(args.mat_path)
    if "X" not in mat:
        raise SystemExit("Missing 'X' key in the .mat file.")

    dims = _load_dims(mat)
    X_np = _reshape_X(np.array(mat["X"]), dims)

    dtype = _dtype_from_str(args.dtype)
    X = torch.tensor(X_np, dtype=dtype)

    rank_true = None
    if "y" in mat:
        y = np.array(mat["y"])
        if y.ndim == 2:
            rank_true = int(y.shape[1])

    if rank_true is None:
        rank_true = 0

    config = {
        "shape": list(X.shape),
        "rank_true": int(rank_true),
        "noise_std": 0.0,
        "corr_rho": 0.0,
        "dtype": _dtype_name(dtype),
    }

    payload = {
        "X": X,
        "X_clean": X.clone(),
        "factors": None,
        "weights": None,
        "config": config,
        "seed": int(args.seed),
        "meta": {
            "ExAx": np.array(mat.get("ExAx", [])).squeeze().tolist(),
            "EmAx": np.array(mat.get("EmAx", [])).squeeze().tolist(),
            "y": np.array(mat.get("y", [])).tolist(),
        },
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)
    print(f"[info] wrote {args.out_path}")


if __name__ == "__main__":
    main()
