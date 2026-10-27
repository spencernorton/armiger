"""Local metadata classification. Model files are downloaded separately."""
from importlib import import_module

__version__ = "3.3.1"
_EXPORTS = {
    "ArmigerStandaloneONNX": "armiger_inference",
    "ArmigerONNXClassifier": "armiger_inference",
    "ArmigerUltraPipeline": "armiger_ultra_pipeline",
    "ArmigerPipeline": "armiger_ultra_pipeline",
    "AnonymityGuard": "armiger_dht_rsi",
    "AnonymityViolationError": "armiger_dht_rsi",
    "ArmigerDHTRSIEngine": "armiger_dht_rsi",
    "ArmigerDHTDBSync": "armiger_dht_db_sync",
    "resolve_opt_in_sharing": "armiger_dht_rsi",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
