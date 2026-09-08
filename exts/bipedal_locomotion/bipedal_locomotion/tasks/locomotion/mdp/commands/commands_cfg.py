import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .body_height_command import BodyHeightCommand
from .gait_command import GaitCommand  # Import the GaitCommand class
from .wheel_support_velocity_command import WheelSupportVelocityCommand


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
    endpoint_fraction: float = 0.0
    """Fraction of resamples assigned equally to the exact minimum and maximum heights."""
    asset_name: str = "robot"
    """Articulation whose base height is tracked for command metrics."""
    height_sensor_name: str = "height_scanner"
    """Ray caster used to measure height relative to the local terrain plane."""


@configclass
class WheelSupportVelocityCommandCfg(UniformVelocityCommandCfg):
    """Mode-balanced velocity command augmented with Wheel support diagnostics."""

    class_type: type = WheelSupportVelocityCommand
    contact_sensor_names: tuple[str, str] = MISSING
    terrain_sensor_names: tuple[str, str] = MISSING
    wheel_body_names: tuple[str, str] = MISSING
    wheel_radius: float = 0.128
    force_off: float = 5.0
    force_on: float = 10.0
    geometry_tolerance_on: float = 0.02
    geometry_tolerance_off: float = 0.035
    support_threshold: float = 0.5
    tracking_std: float = math.sqrt(0.12)
    moving_command_threshold: float = 0.1
    rel_straight_envs: float = 0.30
    """Fraction of environments commanded to translate with zero yaw rate."""
    rel_yaw_only_envs: float = 0.10
    """Fraction of environments commanded to yaw with zero planar velocity."""
    rel_mixed_envs: float = 0.35
    """Fraction of environments commanded to translate and yaw simultaneously."""
