"""Dataset specs for the revision runs.

Spec strings (one per dataset):
  amino=pt:data/amino/*.pt                       a .pt payload {"X", optional "factors", "config"}
  covid=pt:data/covid19/covid19_serology.pt
  kinetic=pt:data/kinetic/X60t.pt               non-finite entries are missing: set to 0, excluded from the error
  synth_default=manifest:data/synthetic/manifest_*.json?noise=0.01&rho=0.5&seed={seed}
  snr10=synth:snr_db=10,rho=0.5,R=6,shape=20x20x20x20,seed={seed}
  sig01=synth:sigma=0.01,rho=0.5,R=6,shape=20x20x20x20,seed=0

The text before "=" is the dataset name used in all outputs. "{seed}" is replaced by the run seed (resolve_spec), so
seed s runs on tensor s, as in the submitted experiments (synthetic_..._seed{s:02d}.pt for seed s).
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.synthetic import make_synthetic

# Settings of the submitted experiments, per dataset kind (t_max, prune_period, ridge, tol, Top-K refit sweeps).
PAPER_SETTINGS = {
    "synthetic": dict(t_max=200, prune_period=40, ridge=1e-6, tol=0.0, topk_refit=0),
    "amino": dict(t_max=200, prune_period=40, ridge=1e-6, tol=0.0, topk_refit=50),
    "kinetic": dict(t_max=200, prune_period=40, ridge=1e-6, tol=0.0, topk_refit=0),
    "covid": dict(t_max=200, prune_period=40, ridge=1e-6, tol=1e-12, topk_refit=0),
}


@dataclass
class Dataset:
    name: str
    kind: str
    X: torch.Tensor                    # zero-filled at missing entries
    Omega: Optional[torch.Tensor]      # 1 = observed, None if fully observed
    gt_factors: Optional[List[torch.Tensor]] = None
    meta: Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.X.shape)

    @property
    def xnorm(self) -> float:
        Xo = self.X if self.Omega is None else self.Omega * self.X
        return float(torch.linalg.norm(Xo).clamp_min(1e-12).item())


def _kind_from_name(name: str, default: str) -> str:
    low = name.lower()
    for k in ("amino", "acid", "kinetic", "x60t", "covid"):
        if k in low:
            return {"acid": "amino", "x60t": "kinetic"}.get(k, k)
    return default


def _sanitize(X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    finite = torch.isfinite(X)
    if bool(finite.all()):
        return X, None
    X = X.clone()
    X[~finite] = 0.0
    return X, finite.to(dtype=X.dtype)


def _floor_from_config(shape, rank_true, rho, sigma) -> Tuple[float, float]:
    """Manuscript Eqs. (snr_energies)-(noise_floor): expected SNR (dB) and eps* for unit weights."""
    N = len(shape)
    sig = rank_true * (1.0 + (rank_true - 1) * rho ** N)
    noi = sigma ** 2 * float(np.prod(shape))
    if noi <= 0:
        return float("inf"), 0.0
    snr = sig / noi
    return 10.0 * math.log10(snr), (1.0 + snr) ** -0.5


def parse_spec(spec: str) -> Tuple[str, str, str]:
    if "=" not in spec.split(":", 1)[0]:
        raise ValueError(f"dataset spec needs a name: NAME=kind:..., got {spec}")
    name, rest = spec.split("=", 1)
    kind, arg = rest.split(":", 1)
    return name.strip(), kind.strip(), arg.strip()


def resolve_spec(spec: str, seed: int) -> str:
    """Put the run seed into a dataset spec that has a {seed} placeholder (no-op for the real tensors)."""
    return spec.replace("{seed}", str(int(seed)))


@lru_cache(maxsize=128)
def load_dataset(spec: str, dtype: str = "float32") -> Dataset:
    if "{seed}" in spec:
        raise ValueError(f"unresolved seed placeholder in {spec}; call resolve_spec(spec, seed) first")
    name, kind, arg = parse_spec(spec)
    dt = {"float32": torch.float32, "float64": torch.float64}[dtype]
    if kind == "pt":
        paths = sorted(glob.glob(arg))
        if not paths:
            raise FileNotFoundError(f"[{name}] no file matches {arg}")
        payload = torch.load(paths[0], map_location="cpu")
        X, Omega = _sanitize(payload["X"].to(dtype=dt))
        gt = payload.get("factors")
        gt = [A.to(dtype=dt) for A in gt] if gt is not None else None
        cfg = payload.get("config", {}) or {}
        meta = {"path": paths[0], "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
                                              if isinstance(v, (int, float, str, list, tuple))}}
        if gt is not None and "noise_std" in cfg and "corr_rho" in cfg:
            meta["snr_db"], meta["floor"] = _floor_from_config(X.shape, int(cfg.get("rank_true", len(gt[0][0]))),
                                                               float(cfg["corr_rho"]), float(cfg["noise_std"]))
            meta["rank_true"] = int(cfg.get("rank_true", gt[0].shape[1]))
        return Dataset(name, _kind_from_name(name, "synthetic" if gt is not None else "amino"), X, Omega, gt, meta)

    if kind == "manifest":
        mpath, _, query = arg.partition("?")
        mfiles = sorted(glob.glob(mpath))
        if not mfiles:
            raise FileNotFoundError(f"[{name}] no manifest matches {mpath}")
        want = dict(kv.split("=") for kv in query.split("&") if kv) if query else {}
        recs = []
        for mf in mfiles:
            for e in json.loads(Path(mf).read_text()):
                p = Path(e["path"])
                if not p.is_absolute() and not p.exists():
                    # manifest paths are relative to the repository root; also accept them next to the manifest
                    cands = [Path(mf).parent / p, Path(mf).parent / p.name]
                    p = next((c for c in cands if c.exists()), cands[0])
                recs.append((p, e))
        def ok(e):
            if "noise" in want and not math.isclose(float(e["noise_std"]), float(want["noise"]), rel_tol=1e-9, abs_tol=1e-15):
                return False
            if "rho" in want and not math.isclose(float(e["corr_rho"]), float(want["rho"]), rel_tol=1e-9, abs_tol=1e-15):
                return False
            if "seed" in want and int(e.get("seed", -1)) != int(want["seed"]):
                return False
            return True
        recs = [r for r in recs if ok(r[1])]
        idx = int(want.get("index", 0))
        if not recs or idx >= len(recs):
            raise FileNotFoundError(f"[{name}] no manifest record matches {query}")
        p, e = recs[idx]
        payload = torch.load(p, map_location="cpu")
        X = payload["X"].to(dtype=dt)
        gt = payload.get("factors")
        gt = [A.to(dtype=dt) for A in gt] if gt is not None else None
        snr_db, floor = _floor_from_config(X.shape, int(e["rank_true"]), float(e["corr_rho"]), float(e["noise_std"]))
        meta = {"path": str(p), "rank_true": int(e["rank_true"]), "rho": float(e["corr_rho"]),
                "sigma": float(e["noise_std"]), "snr_db": snr_db, "floor": floor, "floor_kind": "expected"}
        return Dataset(name, "synthetic", X, None, gt, meta)

    if kind == "synth":
        kv = dict(item.split("=") for item in arg.split(",") if item)
        shape = tuple(int(s) for s in kv.get("shape", "20x20x20x20").split("x"))
        out = make_synthetic(
            shape=shape, rank_true=int(kv.get("R", 6)), rho=float(kv.get("rho", 0.5)),
            snr_db=float(kv["snr_db"]) if "snr_db" in kv else None,
            sigma=float(kv["sigma"]) if "sigma" in kv else None,
            seed=int(kv.get("seed", 0)) + int(kv.get("seed_offset", 0)),
        )
        X = out["X"].to(dtype=dt)
        meta = {k: v for k, v in out.items() if k not in ("X", "X_clean", "factors")}
        meta["floor_kind"] = "realized"
        return Dataset(name, "synthetic", X, None, [A.to(dtype=dt) for A in out["factors"]], meta)

    raise ValueError(f"unknown dataset kind {kind}")


def mu_effective(mu: float, xnorm: float, rank_ref: int) -> float:
    """Manuscript Eq. (8): mu_eff = mu ||X||_F^2 / R."""
    return float(mu) * float(xnorm) ** 2 / float(max(1, int(rank_ref)))
