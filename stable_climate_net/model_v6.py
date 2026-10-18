"""Shared spatial utilities retained from the validated development code."""
from __future__ import annotations
import math
import torch

def pairwise_haversine_km(coords: torch.Tensor) -> torch.Tensor:
    """Symmetric great-circle distance matrix for [S,2] lat/lon degrees."""
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coordinates must have shape [stations,2] = [lat,lon]")
    if not torch.isfinite(coords).all():
        raise ValueError("coordinates contain non-finite values")
    rad = coords * (math.pi / 180.0)
    lat, lon = rad[:, 0], rad[:, 1]
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon / 2).pow(2)
    a = a.clamp(0.0, 1.0)
    return 6371.0088 * 2.0 * torch.asin(torch.sqrt(a))

def operator_errors(sym: torch.Tensor, asym: torch.Tensor):
    """Return exact-structure diagnostics for [...,S,S] operator tensors."""
    se = (sym - sym.transpose(-1, -2)).abs().max()
    ae = (asym + asym.transpose(-1, -2)).abs().max()
    inner = (sym * asym).sum(dim=(-1, -2)).abs().max()
    denom = (sym.pow(2).sum(dim=(-1, -2)).sqrt() * asym.pow(2).sum(dim=(-1, -2)).sqrt()).clamp_min(1e-12)
    rel_inner = ((sym * asym).sum(dim=(-1, -2)).abs() / denom).max()
    return se, ae, inner, rel_inner
