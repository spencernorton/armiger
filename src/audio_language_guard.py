"""Compatibility import; prefer ``armiger.audio_language_guard`` in new code."""
import sys
from armiger import audio_language_guard as _implementation
sys.modules[__name__] = _implementation
