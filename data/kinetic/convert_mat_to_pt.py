#!/usr/bin/env python3
"""
Convert X60t.mat to a .pt payload compatible with synthetic datasets.
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

EXPECTED_SHAPE: Tuple[int, ...] = (64, 12, 10, 60)


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


def _reshape_X(X: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
    if X.shape == expected_shape:
        return X
    if X.ndim == 2 and int(X.size) == int(np.prod(expected_shape)):
        # MATLAB column-major layout: use Fortran order to recover original tensor.
        return np.reshape(X, expected_shape, order="F")
    raise ValueError(f"Unexpected X shape {X.shape}; expected {expected_shape}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert X60t.mat to a .pt dataset payload.")
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=Path("data/kinetic/X60t.mat"),
        help="Path to X60t.mat.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/kinetic/X60t.pt"),
        help="Output .pt path.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Output tensor dtype.")
    parser.add_argument("--seed", type=int, default=0, help="Seed stored in payload metadata.")
    args = parser.parse_args()

    mat: Dict = sio.loadmat(args.mat_path)
    if "X60t" not in mat:
        raise SystemExit("Missing 'X60t' key in the .mat file.")

    X_np = _reshape_X(np.array(mat["X60t"]), EXPECTED_SHAPE)
    if X_np.shape != EXPECTED_SHAPE:
        raise ValueError(f"X60t tensor has shape {X_np.shape}, expected {EXPECTED_SHAPE}.")

    dtype = _dtype_from_str(args.dtype)
    X = torch.tensor(X_np, dtype=dtype)

    config = {
        "shape": list(X.shape),
        "rank_true": 0,
        "noise_std": 0.0,
        "corr_rho": 0.0,
        "dtype": _dtype_name(dtype),
    }

    print(config)

    payload = {
        "X": X,
        "X_clean": X.clone(),
        "factors": None,
        "weights": None,
        "config": config,
        "seed": int(args.seed),
        "meta": {
            "ExAx": [],
            "EmAx": [],
            "y": [],
        },
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)
    print(f"[info] wrote {args.out_path}")


if __name__ == "__main__":
    main()
