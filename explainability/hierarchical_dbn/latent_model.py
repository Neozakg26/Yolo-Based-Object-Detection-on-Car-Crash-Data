"""
Latent factor definitions for the hierarchical DBN.

Defines the three-level hierarchy:
    - Observable features (discrete versions of tracking outputs)
    - Intermediate latent factors
    - Top-level accident risk
"""

from enum import Enum
from typing import Dict, List


class LatentFactors(Enum):
    """Intermediate latent factors in the hierarchical DBN."""
    COLLISION_IMMINENCE = "collision_imminence"
    BEHAVIOURAL_RISK = "behavioural_risk"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"


class RiskLevel(Enum):
    """Top-level accident risk states."""
    SAFE = 0
    ELEVATED = 1
    CRITICAL = 2

    @classmethod
    def from_tta(cls, time_to_accident: float, fps: float = 10.0) -> 'RiskLevel':
        """
        Assign risk level based on time-to-accident.

        Args:
            time_to_accident: Seconds until accident
            fps: Frames per second of video

        Returns:
            RiskLevel based on TTA thresholds from proposal
        """
        if time_to_accident >= 2.5:
            return cls.SAFE
        elif time_to_accident >= 1.5:
            return cls.ELEVATED
        else:
            return cls.CRITICAL


class LatentState(Enum):
    """States for intermediate latent factors."""
    LOW = 0
    MODERATE = 1
    HIGH = 2


# Mapping from latent factors to their parent observables
# These observables will be discretized versions (suffix _d)
# Updated to include new tracker features for improved collision risk assessment
OBSERVABLE_TO_LATENT: Dict[LatentFactors, List[str]] = {
    LatentFactors.COLLISION_IMMINENCE: [
        # TTC metrics (multiple variants for robustness)
        "ttc_proxy_d",           # Original TTC estimate
        "ttc_relative_d",        # NEW: TTC using relative velocity (more accurate)
        "ttc_smoothed_d",        # NEW: Temporally smoothed TTC (less noisy)
        "ttc_rate_d",            # Rate of TTC change (danger trend)
        # Approach dynamics
        "closing_rate_d",        # Absolute closing rate
        "rel_closing_rate_d",    # NEW: Relative closing rate (accounts for ego motion)
        "proximity_rate_d",      # NEW: Rate of proximity change
        # Distance metrics
        "proximity_d",           # Current distance proxy
        "min_distance_t_d",      # Environment: closest object
    ],
    LatentFactors.BEHAVIOURAL_RISK: [
        # Absolute velocity
        "vx_d",                  # Lateral velocity
        "vy_d",                  # Longitudinal velocity
        "speed_d",               # Speed magnitude
        # NEW: Relative velocity (object relative to ego)
        "rel_vx_d",              # Relative lateral velocity
        "rel_vy_d",              # Relative longitudinal velocity
        "rel_speed_d",           # Relative speed magnitude
        # Acceleration (indicates intent/reaction)
        "ax_d",                  # Lateral acceleration
        "ay_d",                  # Longitudinal acceleration
        # Ego vehicle state
        "ego_speed_d",           # Ego vehicle speed
        "ego_accel_d",           # Ego braking/acceleration
        # Risk speed metrics
        "risk_speed_d",          # Absolute risk speed
        "rel_risk_speed_d",      # NEW: Relative risk speed
    ],
    LatentFactors.ENVIRONMENTAL_HAZARD: [
        "num_objects_close_t_d", # Scene density
        "mean_rel_speed_t_d",    # Average relative motion in scene
        "min_ttc_t_d",           # Most dangerous object TTC
    ],
}

# Reverse mapping: observable -> latent factor
LATENT_PARENTS: Dict[str, LatentFactors] = {
    obs: latent
    for latent, observables in OBSERVABLE_TO_LATENT.items()
    for obs in observables
}

# All observable names (discretized versions)
ALL_OBSERVABLES: List[str] = [
    obs for observables in OBSERVABLE_TO_LATENT.values() for obs in observables
]

# State cardinalities
OBSERVABLE_CARDINALITY: Dict[str, int] = {
    # =====================================================
    # COLLISION IMMINENCE OBSERVABLES
    # =====================================================
    # TTC metrics
    "ttc_proxy_d": 4,           # Critical, Warning, Caution, Safe
    "ttc_relative_d": 4,        # NEW: Same states as ttc_proxy
    "ttc_smoothed_d": 4,        # NEW: Same states as ttc_proxy
    "ttc_rate_d": 4,            # Rapid_Decrease, Decreasing, Stable, Increasing
    # Approach dynamics
    "closing_rate_d": 3,        # Retreating, Approaching_Slow, Approaching_Fast
    "rel_closing_rate_d": 3,    # NEW: Same states as closing_rate
    "proximity_rate_d": 3,      # NEW: Distancing, Stable, Approaching
    # Distance metrics
    "proximity_d": 3,           # Far, Medium, Close
    "min_distance_t_d": 3,      # Danger, Caution, Safe

    # =====================================================
    # BEHAVIOURAL RISK OBSERVABLES
    # =====================================================
    # Absolute velocity
    "vx_d": 3,                  # Fast_Left, Neutral, Fast_Right
    "vy_d": 3,                  # Receding, Neutral, Approaching
    "speed_d": 3,               # Slow, Moderate, Fast
    "heading_d": 4,             # NEW: Left, Down, Up, Right
    # NEW: Relative velocity (object relative to ego)
    "rel_vx_d": 3,              # Fast_Left, Neutral, Fast_Right
    "rel_vy_d": 3,              # Receding, Neutral, Approaching
    "rel_speed_d": 3,           # Slow, Moderate, Fast
    "rel_heading_d": 4,         # NEW: Left, Down, Up, Right
    # Acceleration
    "ax_d": 3,                  # Decel_Left, Steady, Accel_Right
    "ay_d": 3,                  # Decel_Away, Steady, Accel_Toward
    # Ego vehicle
    "ego_speed_d": 3,           # Slow, Normal, Fast
    "ego_accel_d": 3,           # Braking, Steady, Accelerating
    # Risk speed
    "risk_speed_d": 3,          # Low, Medium, High
    "rel_risk_speed_d": 3,      # NEW: Low, Medium, High

    # =====================================================
    # ENVIRONMENTAL HAZARD OBSERVABLES
    # =====================================================
    "num_objects_close_t_d": 3, # Sparse, Moderate, Crowded
    "mean_rel_speed_t_d": 3,    # Low, Medium, High
    "min_ttc_t_d": 3,           # Critical, Warning, Safe

    # =====================================================
    # TRACK CONFIDENCE (optional, for weighting)
    # =====================================================
    "pos_uncertainty_d": 3,     # NEW: High_Confidence, Medium_Confidence, Low_Confidence
    "vel_uncertainty_d": 3,     # NEW: High_Confidence, Medium_Confidence, Low_Confidence
}

LATENT_CARDINALITY: Dict[str, int] = {
    LatentFactors.COLLISION_IMMINENCE.value: 3,  # Low, Moderate, High
    LatentFactors.BEHAVIOURAL_RISK.value: 3,     # Low, Moderate, High
    LatentFactors.ENVIRONMENTAL_HAZARD.value: 3, # Low, Moderate, High
    "accident_risk": 3,                           # Safe, Elevated, Critical
}


def get_all_variable_names() -> List[str]:
    """Get all variable names in the DBN (observables + latents + risk)."""
    return (
        ALL_OBSERVABLES +
        [lf.value for lf in LatentFactors] +
        ["accident_risk"]
    )


def get_variable_cardinality(var_name: str) -> int:
    """Get the number of states for a variable."""
    if var_name in OBSERVABLE_CARDINALITY:
        return OBSERVABLE_CARDINALITY[var_name]
    elif var_name in LATENT_CARDINALITY:
        return LATENT_CARDINALITY[var_name]
    else:
        raise ValueError(f"Unknown variable: {var_name}")
