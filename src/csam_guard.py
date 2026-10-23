"""Compatibility import; prefer ``armiger.csam_guard`` in new code."""
import sys
from armiger import csam_guard as _implementation
sys.modules[__name__] = _implementation
