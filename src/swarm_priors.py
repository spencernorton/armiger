"""Compatibility import; prefer ``armiger.swarm_priors`` in new code."""
import sys
from armiger import swarm_priors as _implementation
sys.modules[__name__] = _implementation
