"""Locked public SymmClimateNet architecture."""
from __future__ import annotations
from stable_climate_net.model_v7 import SetTemporalSymmetricSpatialNet
from stable_climate_net.v3_locked import LOCKED_SPEC, locked_spec_dict

class SymmClimateNet(SetTemporalSymmetricSpatialNet):
    """Symmetry-Guided Spatio-Temporal Climate Network."""
    def __init__(self, *, n_features: int, station_coordinates,
                 target_feature_index: int = 0, history: int = 30,
                 hidden: int = 48, head_hidden: int = 64,
                 message_dim: int = 12, n_heads: int = 4,
                 aux_hidden: int = 32, quantiles=(0.05, 0.50, 0.95)):
        super().__init__(
            n_features=n_features, station_coordinates=station_coordinates,
            history=history, hidden=hidden, head_hidden=head_hidden,
            message_dim=message_dim, n_heads=n_heads, aux_hidden=aux_hidden,
            quantiles=quantiles, target_feature_index=target_feature_index,
            mode="set_temporal_plus_symmetric_spatial",
        )

def build_model(*, n_features: int, station_coordinates, target_feature_index: int = 0):
    s = LOCKED_SPEC
    return SymmClimateNet(
        n_features=n_features, station_coordinates=station_coordinates,
        target_feature_index=target_feature_index, history=s.history,
        hidden=s.hidden, head_hidden=s.head_hidden, message_dim=s.message_dim,
        n_heads=s.n_heads, aux_hidden=s.aux_hidden, quantiles=s.quantiles,
    )
