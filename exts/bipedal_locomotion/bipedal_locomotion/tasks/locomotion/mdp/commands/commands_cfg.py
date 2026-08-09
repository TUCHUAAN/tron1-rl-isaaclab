import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .body_height_command import BodyHeightCommand
from .gait_command import GaitCommand  # Import the GaitCommand class


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""


@configclass
class UniformBodyHeightCommandCfg(CommandTermCfg):
    """Configuration for a smooth body-height command relative to local terrain."""

    class_type: type = BodyHeightCommand

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Commanded relative base-height range in metres."""

    ranges: Ranges = MISSING
    resampling_time_range: tuple[float, float] = (5.0, 8.0)
    max_rate: float = 0.08
    """Maximum command slew rate in metres per second."""
