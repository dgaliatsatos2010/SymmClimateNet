"""Locked V3 architecture selected by the completed V3rc3 Stage-B protocol."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .model_v7 import SetTemporalSymmetricSpatialNet

LOCK_STAGE = "V3rc3_StageB_ModelLock"
LOCK_DECISION = "LOCK_SET_TEMPORAL_PLUS_SYMMETRIC_SPATIAL"
LOCKED_VARIANT = "set_temporal_plus_symmetric_spatial"
LOCK_REASON = "operational_tie_resolved_by_parsimony"

@dataclass(frozen=True)
class V3LockedSpec:
    variant: str = LOCKED_VARIANT
    history: int = 30
    hidden: int = 48
    head_hidden: int = 64
    message_dim: int = 12
    n_heads: int = 4
    aux_hidden: int = 32
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    lambda_stability: float = 0.02
    raw_epochs: int = 35
    raw_patience: int = 7
    spatial_epochs: int = 30
    spatial_patience: int = 6
    batch_size: int = 64
    learning_rate: float = 1e-3

LOCKED_SPEC = V3LockedSpec()

def locked_spec_dict() -> dict:
    d = asdict(LOCKED_SPEC)
    d.update({"lock_stage": LOCK_STAGE,"lock_decision": LOCK_DECISION,"lock_reason": LOCK_REASON})
    return d

def build_locked_v3_model(*, n_features: int, station_coordinates, target_feature_index: int):
    s = LOCKED_SPEC
    return SetTemporalSymmetricSpatialNet(
        n_features=n_features,
        station_coordinates=station_coordinates,
        history=s.history,
        hidden=s.hidden,
        head_hidden=s.head_hidden,
        message_dim=s.message_dim,
        n_heads=s.n_heads,
        aux_hidden=s.aux_hidden,
        quantiles=s.quantiles,
        target_feature_index=int(target_feature_index),
        mode=LOCKED_VARIANT,
    )
