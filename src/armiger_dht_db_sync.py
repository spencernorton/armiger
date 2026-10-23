"""Compatibility import; prefer ``armiger.armiger_dht_db_sync`` in new code."""
import sys
from armiger import armiger_dht_db_sync as _implementation
sys.modules[__name__] = _implementation
