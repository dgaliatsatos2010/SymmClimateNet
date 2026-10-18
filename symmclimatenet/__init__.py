"""Public API for SymmClimateNet v1.0.1."""
from .model import SymmClimateNet, build_model, locked_spec_dict
__version__ = "1.0.1"
__all__ = ["SymmClimateNet", "build_model", "locked_spec_dict", "__version__"]
