"""Compatibility import; prefer ``armiger.armiger_ultra_pipeline`` in new code."""
import sys
from armiger import armiger_ultra_pipeline as _implementation
sys.modules[__name__] = _implementation
