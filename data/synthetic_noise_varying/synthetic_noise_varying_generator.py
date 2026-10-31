#!/usr/bin/env python3
"""
Generate synthetic CP datasets across multiple noise levels and random seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from src.utils.tensor import cp_reconstruct


@dataclass
class DatasetConfig:
    shape: Tuple[int, ...]
    rank_true: int
    noise_std: float
    corr_rho: float
    dtype: str

    def tag(self) -> str:
        dims = "x".join(str(s) for s in self.shape)
        return f"I{dims}_R{self.rank_true}_rho{self.corr_rho}_noise{self.noise_std}"


def _parse_shape(raw: str) -> Tuple[int, ...]:
    cleaned = raw.lower().replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain only integers") from exc
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return dims


def _parse_seeds(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("seeds list cannot be empty")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc


def _parse_noise_stds(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("noise list cannot be empty")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("noise values must be floats") from exc
    if any(v < 0.0 for v in values):
        raise argparse.ArgumentTypeError("noise values must be non-negative")
    return values


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
        raise argparse.ArgumentTypeError(
            f"unsupported dtype '{raw}'. Use one of: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[key]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _noise_folder_name(noise_std: float) -> str:
    return f"noise_{noise_std}"


def generate_cp_tensor(
    shape: Sequence[int],
    rank_true: int,
    noise_std: float,
    corr_rho: float,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    factors: List[torch.Tensor] = []
    for I in shape:
        shared = torch.randn((I, 1), generator=g, dtype=dtype)
        base = torch.randn((I, rank_true), generator=g, dtype=dtype)
        alpha = math.sqrt(max(0.0, 1.0 - corr_rho))
        beta = math.sqrt(max(0.0, corr_rho))
        A = alpha * base + beta * shared
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
        factors.append(A)

    weights = torch.ones((rank_true,), dtype=dtype)
    X_clean = cp_reconstruct(factors, weights)
    noise = noise_std * torch.randn(X_clean.shape, generator=g, dtype=dtype)
    X = X_clean + noise
    return X, X_clean, factors, weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CP tensors across multiple noise levels."
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=(20, 20, 20, 20),
        help="Tensor shape, e.g. '20,20,20,20' or '20x20x20x20'.",
    )
    parser.add_argument("--rank-true", type=int, default=6, help="True CP rank.")
    parser.add_argument(
        "--noise-stds",
        type=_parse_noise_stds,
        default=[0.0, 0.001, 0.01, 0.05, 0.1],
        help="Comma-separated list of Gaussian noise stds.",
    )
    parser.add_argument(
        "--corr-rho",
        type=float,
        default=0.5,
        help="Mode-wise collinearity control in [0, 1].",
    )
    parser.add_argument("--dtype", type=_dtype_from_str, default=torch.float32)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of seeds to generate per noise level (ignored if --seeds is provided).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed when using --num-seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=None,
        help="Comma-separated list of explicit seeds (overrides --num-seeds).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic_noise_varying"),
        help="Output directory for saved datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset files if they already exist.",
    )
    args = parser.parse_args()

    seeds = (
        args.seeds
        if args.seeds is not None
        else list(range(args.seed_start, args.seed_start + args.num_seeds))
    )
    if not seeds:
        raise SystemExit("No seeds provided. Use --num-seeds or --seeds.")

    noise_stds = list(args.noise_stds)
    if not noise_stds:
        raise SystemExit("No noise levels provided. Use --noise-stds.")

    dtype = args.dtype
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    width = max(1, len(str(max(seeds))))

    for noise_std in noise_stds:
        cfg = DatasetConfig(
            shape=tuple(args.shape),
            rank_true=args.rank_true,
            noise_std=noise_std,
            corr_rho=args.corr_rho,
            dtype=_dtype_name(dtype),
        )
        noise_dir = out_dir / _noise_folder_name(noise_std)
        noise_dir.mkdir(parents=True, exist_ok=True)

        tag = cfg.tag()
        manifest: List[dict] = []

        for seed in seeds:
            X, X_clean, factors, weights = generate_cp_tensor(
                shape=cfg.shape,
                rank_true=cfg.rank_true,
                noise_std=cfg.noise_std,
                corr_rho=cfg.corr_rho,
                dtype=dtype,
                seed=seed,
            )
            seed_str = f"{seed:0{width}d}"
            filename = f"synthetic_{tag}_seed{seed_str}.pt"
            path = noise_dir / filename
            if path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite existing file: {path}")
            payload = {
                "X": X.cpu(),
                "X_clean": X_clean.cpu(),
                "factors": [A.cpu() for A in factors],
                "weights": weights.cpu(),
                "config": asdict(cfg),
                "seed": seed,
            }
            torch.save(payload, path)
            manifest.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "shape": list(cfg.shape),
                    "rank_true": cfg.rank_true,
                    "noise_std": cfg.noise_std,
                    "corr_rho": cfg.corr_rho,
                    "dtype": cfg.dtype,
                }
            )

        manifest_path = noise_dir / f"manifest_{tag}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
