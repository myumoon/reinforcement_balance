"""Compatibility alias for the shared artifact store implementation."""

from __future__ import annotations

import sys

from reinbalance_survivors_contracts import artifact_store as _shared

sys.modules[__name__] = _shared
