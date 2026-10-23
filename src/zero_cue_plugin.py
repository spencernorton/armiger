"""Compatibility import; prefer ``armiger.zero_cue_plugin`` in new code."""
import sys
from armiger import zero_cue_plugin as _implementation
sys.modules[__name__] = _implementation
