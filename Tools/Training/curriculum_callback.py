"""Compatibility wrapper for the Survivors curriculum callback.

New code should use ``games.survivors.survivors_curriculum`` through the
game config hooks instead of importing this module directly.
"""

from games.survivors.survivors_curriculum import CurriculumCallback, PHASES

__all__ = ["CurriculumCallback", "PHASES"]
