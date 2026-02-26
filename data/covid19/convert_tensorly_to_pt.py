#!/usr/bin/env python3
"""
Convert the TensorLy COVID-19 serology dataset into a .pt payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import torch

try:
    import tensorly as tl
    from tensorly import datasets as tl_datasets
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("tensorly is required to load the dataset. Please install tensorly and retry.") from exc


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


def _resolve_loader(name: str) -> Callable[[], Any]:
    candidates = []
    if name:
        candidates.append(name)
    if name and not name.startswith("load_"):
        candidates.append(f"load_{name}")
    if name and name.startswith("load_"):
        candidates.append(name[len("load_") :])

    for candidate in candidates:
        if hasattr(tl_datasets, candidate):
            fn = getattr(tl_datasets, candidate)
            if callable(fn):
                return fn

    covid_candidates = sorted(
        n
        for n in dir(tl_datasets)
        if "covid" in n.lower() and callable(getattr(tl_datasets, n))
    )
    hint = f" Available loaders: {', '.join(covid_candidates)}" if covid_candidates else ""
    raise SystemExit(f"Dataset loader '{name}' not found in tensorly.datasets.{hint}")


def _is_tensor_like(obj: Any) -> bool:
    if isinstance(obj, np.ndarray):
        return True
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return True
    return False


def _select_tensor_from_mapping(mapping: Dict[str, Any]) -> Tuple[str, Any]:
    preferred = ("tensor", "data", "X", "x")
    for key in preferred:
        if key in mapping and _is_tensor_like(mapping[key]):
            return key, mapping[key]
    for key, value in mapping.items():
        if _is_tensor_like(value):
            return key, value
    raise ValueError("No tensor-like entry found in dataset mapping.")


def _select_tensor_from_sequence(seq: Iterable[Any]) -> Tuple[int, Any]:
    for idx, value in enumerate(seq):
        if _is_tensor_like(value):
            return idx, value
    raise ValueError("No tensor-like entry found in dataset sequence.")


def _select_tensor_from_object(obj: Any) -> Tuple[str, Any]:
    preferred = ("tensor", "data", "X", "x")
    for attr in preferred:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if _is_tensor_like(value):
                return attr, value
    for attr, value in vars(obj).items():
        if _is_tensor_like(value):
            return attr, value
    raise ValueError("No tensor-like entry found in dataset object.")


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "values"):
        tensor = tensor.values
    try:
        return tl.to_numpy(tensor)
    except Exception:
        return np.asarray(tensor)


def _serialize_meta(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize_meta(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_meta(v) for k, v in value.items()}
    return value


def _extract_tensor_and_meta(dataset: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(dataset, dict):
        key, tensor = _select_tensor_from_mapping(dataset)
        meta = {k: v for k, v in dataset.items() if k != key}
    elif isinstance(dataset, (list, tuple)):
        idx, tensor = _select_tensor_from_sequence(dataset)
        remainder = [v for i, v in enumerate(dataset) if i != idx]
        if len(remainder) == 1 and isinstance(remainder[0], dict):
            meta = dict(remainder[0])
        else:
            meta = {"extra": remainder}
    else:
        attr, tensor = _select_tensor_from_object(dataset)
        meta = {k: v for k, v in vars(dataset).items() if k != attr}

    meta = {str(k): _serialize_meta(v) for k, v in meta.items() if not callable(v)}
    X_np = _to_numpy(tensor)
    return X_np, meta


def _infer_rank_true(meta: Dict[str, Any]) -> int:
    for key in ("rank_true", "rank", "true_rank"):
        value = meta.get(key)
        if isinstance(value, (int, np.integer)):
            return int(value)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the TensorLy COVID-19 serology dataset to a .pt dataset payload."
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="covid19_serology",
        help="TensorLy dataset loader name (e.g., covid19_serology or load_covid19_serology).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/covid19/covid19_serology.pt"),
        help="Output .pt path.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Output tensor dtype.")
    parser.add_argument("--seed", type=int, default=0, help="Seed stored in payload metadata.")
    parser.add_argument(
        "--rank-true",
        type=int,
        default=None,
        help="Optional true rank to store in config (default: infer or 0).",
    )
    args = parser.parse_args()

    loader = _resolve_loader(args.loader)
    dataset = loader()

    X_np, meta = _extract_tensor_and_meta(dataset)
    if X_np.ndim < 2:
        raise SystemExit(f"Expected tensor with ndim >= 2, got shape {X_np.shape}.")

    dtype = _dtype_from_str(args.dtype)
    X = torch.tensor(X_np, dtype=dtype)
    rank_true = args.rank_true if args.rank_true is not None else _infer_rank_true(meta)

    config = {
        "shape": list(X.shape),
        "rank_true": int(rank_true),
        "noise_std": 0.0,
        "corr_rho": 0.0,
        "dtype": _dtype_name(dtype),
    }

    print(config)

    meta = dict(meta)
    meta["tensorly_loader"] = loader.__name__

    payload = {
        "X": X,
        "X_clean": X.clone(),
        "factors": None,
        "weights": None,
        "config": config,
        "seed": int(args.seed),
        "meta": meta,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)
    print(f"[info] wrote {args.out_path}")


if __name__ == "__main__":
    main()
