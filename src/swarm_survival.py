"""Compatibility import; prefer ``armiger.swarm_survival`` in new code."""
import sys
from armiger import swarm_survival as _implementation
sys.modules[__name__] = _implementation
