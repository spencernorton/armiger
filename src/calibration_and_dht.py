"""Compatibility import; prefer ``armiger.calibration_and_dht`` in new code."""
import sys
from armiger import calibration_and_dht as _implementation
sys.modules[__name__] = _implementation
