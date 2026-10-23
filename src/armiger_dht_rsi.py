"""Compatibility import; prefer ``armiger.armiger_dht_rsi`` in new code."""
import sys
from armiger import armiger_dht_rsi as _implementation
sys.modules[__name__] = _implementation
