"""Compatibility import; prefer ``armiger.armiger_inference`` in new code."""
import sys
from armiger import armiger_inference as _implementation
sys.modules[__name__] = _implementation
