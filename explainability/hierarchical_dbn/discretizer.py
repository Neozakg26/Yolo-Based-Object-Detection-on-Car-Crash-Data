"""
Discretizer for converting continuous tracking features to discrete states.

Uses domain-informed thresholds for risk-relevant features and
percentile-based thresholds for data-driven features.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class DiscretizationConfig:
    """Configuration for discretizing a single feature."""
    bins: List[float]          # Threshold values (ascending)
    labels: List[str]          # State labels (len = len(bins) + 1)
    use_percentiles: bool = False  # If True, bins are percentiles to compute
    clip_outliers: bool = True     # Clip to min/max bin edges


# Default discretization configurations based on domain knowledge
DEFAULT_CONFIGS: Dict[str, DiscretizationConfig] = {
    # =====================================================
    # TIME-TO-COLLISION METRICS
    # =====================================================
    # Original TTC proxy: Critical safety metric
    "ttc_proxy": DiscretizationConfig(
        bins=[1.5, 3.0, 5.0],
        labels=["Critical", "Warning", "Caution", "Safe"],
        use_percentiles=False,
    ),

    # NEW: Improved TTC using relative velocity (from tracker)
    "ttc_relative": DiscretizationConfig(
        bins=[1.5, 3.0, 5.0],
        labels=["Critical", "Warning", "Caution", "Safe"],
        use_percentiles=False,
    ),

    # NEW: Smoothed TTC (temporal filtered)
    "ttc_smoothed": DiscretizationConfig(
        bins=[1.5, 3.0, 5.0],
        labels=["Critical", "Warning", "Caution", "Safe"],
        use_percentiles=False,
    ),

    # TTC rate of change: Negative = danger increasing
    "ttc_rate": DiscretizationConfig(
        bins=[-1.0, 0.0, 1.0],
        labels=["Rapid_Decrease", "Decreasing", "Stable", "Increasing"],
        use_percentiles=False,
    ),

    # =====================================================
    # CLOSING/APPROACH DYNAMICS
    # =====================================================
    # Closing rate (absolute): Positive = approaching
    "closing_rate": DiscretizationConfig(
        bins=[0.0, 5.0],
        labels=["Retreating", "Approaching_Slow", "Approaching_Fast"],
        use_percentiles=False,
    ),

    # NEW: Relative closing rate (accounts for ego motion)
    "rel_closing_rate": DiscretizationConfig(
        bins=[0.0, 5.0],
        labels=["Retreating", "Approaching_Slow", "Approaching_Fast"],
        use_percentiles=False,
    ),

    # NEW: Proximity rate of change (positive = getting closer)
    "proximity_rate": DiscretizationConfig(
        bins=[-0.01, 0.01],
        labels=["Distancing", "Stable", "Approaching"],
        use_percentiles=False,
    ),

    # =====================================================
    # DISTANCE/PROXIMITY METRICS
    # =====================================================
    # Proximity: Scale-depth based (higher = closer)
    "proximity": DiscretizationConfig(
        bins=[0.08, 0.15],
        labels=["Far", "Medium", "Close"],
        use_percentiles=False,
    ),

    # Minimum distance in frame (environment feature)
    "min_distance_t": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Danger", "Caution", "Safe"],
        use_percentiles=True,
    ),

    # =====================================================
    # ABSOLUTE VELOCITY (object in frame)
    # =====================================================
    "vx": DiscretizationConfig(
        bins=[-3.0, 3.0],
        labels=["Fast_Left", "Neutral", "Fast_Right"],
        use_percentiles=False,
    ),

    "vy": DiscretizationConfig(
        bins=[-2.0, 2.0],
        labels=["Receding", "Neutral", "Approaching"],
        use_percentiles=False,
    ),

    # Speed magnitude (scalar velocity)
    "speed": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Slow", "Moderate", "Fast"],
        use_percentiles=True,
    ),

    # NEW: Heading angle (direction of motion)
    "heading": DiscretizationConfig(
        bins=[-1.57, 0.0, 1.57],  # -pi/2, 0, pi/2
        labels=["Left", "Down", "Up", "Right"],
        use_percentiles=False,
    ),

    # =====================================================
    # NEW: RELATIVE VELOCITY (object relative to ego)
    # =====================================================
    "rel_vx": DiscretizationConfig(
        bins=[-3.0, 3.0],
        labels=["Fast_Left", "Neutral", "Fast_Right"],
        use_percentiles=False,
    ),

    "rel_vy": DiscretizationConfig(
        bins=[-2.0, 2.0],
        labels=["Receding", "Neutral", "Approaching"],
        use_percentiles=False,
    ),

    "rel_speed": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Slow", "Moderate", "Fast"],
        use_percentiles=True,
    ),

    "rel_heading": DiscretizationConfig(
        bins=[-1.57, 0.0, 1.57],
        labels=["Left", "Down", "Up", "Right"],
        use_percentiles=False,
    ),

    # =====================================================
    # ACCELERATION
    # =====================================================
    "ax": DiscretizationConfig(
        bins=[-2.0, 2.0],
        labels=["Decel_Left", "Steady", "Accel_Right"],
        use_percentiles=False,
    ),

    "ay": DiscretizationConfig(
        bins=[-2.0, 2.0],
        labels=["Decel_Away", "Steady", "Accel_Toward"],
        use_percentiles=False,
    ),

    # =====================================================
    # EGO VEHICLE METRICS
    # =====================================================
    "ego_speed": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Slow", "Normal", "Fast"],
        use_percentiles=True,
    ),

    "ego_accel": DiscretizationConfig(
        bins=[-2.0, 2.0],
        labels=["Braking", "Steady", "Accelerating"],
        use_percentiles=False,
    ),

    # =====================================================
    # RISK METRICS
    # =====================================================
    # Risk speed (absolute)
    "risk_speed": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Low", "Medium", "High"],
        use_percentiles=True,
    ),

    # NEW: Relative risk speed (accounts for ego motion)
    "rel_risk_speed": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Low", "Medium", "High"],
        use_percentiles=True,
    ),

    # =====================================================
    # ENVIRONMENT/SCENE FEATURES
    # =====================================================
    "num_objects_close_t": DiscretizationConfig(
        bins=[1.0, 3.0],
        labels=["Sparse", "Moderate", "Crowded"],
        use_percentiles=False,
    ),

    "mean_rel_speed_t": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["Low", "Medium", "High"],
        use_percentiles=True,
    ),

    "min_ttc_t": DiscretizationConfig(
        bins=[2.0, 5.0],
        labels=["Critical", "Warning", "Safe"],
        use_percentiles=False,
    ),

    # =====================================================
    # NEW: TRACK CONFIDENCE/UNCERTAINTY
    # =====================================================
    "pos_uncertainty": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["High_Confidence", "Medium_Confidence", "Low_Confidence"],
        use_percentiles=True,
    ),

    "vel_uncertainty": DiscretizationConfig(
        bins=[0.33, 0.67],
        labels=["High_Confidence", "Medium_Confidence", "Low_Confidence"],
        use_percentiles=True,
    ),
}


class ObservableDiscretizer:
    """
    Discretizes continuous tracking features into categorical states for DBN.

    Supports both fixed-threshold and percentile-based discretization.
    """

    def __init__(self, config: Optional[Dict[str, DiscretizationConfig]] = None):
        """
        Initialize discretizer with configuration.

        Args:
            config: Dict mapping feature names to DiscretizationConfig.
                   If None, uses DEFAULT_CONFIGS.
        """
        self.config = config or DEFAULT_CONFIGS.copy()
        self._fitted_bins: Dict[str, List[float]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'ObservableDiscretizer':
        """
        Learn percentile-based thresholds from data.

        Args:
            df: DataFrame with continuous features

        Returns:
            self for chaining
        """
        for feature_name, cfg in self.config.items():
            if feature_name not in df.columns:
                continue

            if cfg.use_percentiles:
                # Convert percentile thresholds to actual values
                values = df[feature_name].dropna()
                if len(values) == 0:
                    self._fitted_bins[feature_name] = cfg.bins
                    continue

                percentile_bins = [p * 100 for p in cfg.bins]
                actual_bins = np.percentile(values, percentile_bins).tolist()
                self._fitted_bins[feature_name] = actual_bins
            else:
                # Use fixed thresholds
                self._fitted_bins[feature_name] = cfg.bins

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert continuous values to discrete states.

        Args:
            df: DataFrame with continuous features

        Returns:
            DataFrame with discretized features (suffix _d)
        """
        if not self._fitted:
            raise RuntimeError("Discretizer must be fitted before transform. Call fit() first.")

        result = df.copy()

        for feature_name, cfg in self.config.items():
            if feature_name not in df.columns:
                continue

            bins = self._fitted_bins.get(feature_name, cfg.bins)
            
            unique_bins = list(dict.fromkeys(bins))
            if len(unique_bins) < len (bins):
                unique_bins.append(unique_bins[0]+0.1)
                
            
            # Create bin edges including -inf and +inf
            bin_edges = [-np.inf] + unique_bins + [np.inf]

            # Discretize
            discrete_col = pd.cut(
                df[feature_name],
                bins=bin_edges,
                labels=cfg.labels,
                include_lowest=True,
            )

            # Add as new column with _d suffix
            result[f"{feature_name}_d"] = discrete_col

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Get list of discretized feature names."""
        return [f"{name}_d" for name in self.config.keys()]

    def discretize_single(
        self,
        feature_name: str,
        value: float
    ) -> str:
        """
        Discretize a single value for a specific feature.

        Args:
            feature_name: Name of the feature (without _d suffix)
            value: Continuous value to discretize

        Returns:
            Discrete state label
        """
        if feature_name not in self.config:
            raise ValueError(f"Unknown feature: {feature_name}")

        cfg = self.config[feature_name]
        bins = self._fitted_bins.get(feature_name, cfg.bins)

        # Find which bin the value falls into
        for i, threshold in enumerate(bins):
            if value < threshold:
                return cfg.labels[i]

        return cfg.labels[-1]

    def discretize_dict(
        self,
        observations: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Discretize a dictionary of observations.

        Args:
            observations: Dict mapping feature names to continuous values

        Returns:
            Dict mapping discretized feature names to state labels
        """
        result = {}
        for feature_name, value in observations.items():
            if feature_name in self.config:
                discrete_name = f"{feature_name}_d"
                result[discrete_name] = self.discretize_single(feature_name, value)
        return result

    def get_state_index(self, feature_name: str, state_label: str) -> int:
        """
        Get the integer index for a state label.

        Args:
            feature_name: Feature name (with or without _d suffix)
            state_label: State label string

        Returns:
            Integer index of the state
        """
        # Remove _d suffix if present
        base_name = feature_name.rstrip("_d") if feature_name.endswith("_d") else feature_name

        if base_name not in self.config:
            raise ValueError(f"Unknown feature: {base_name}")

        cfg = self.config[base_name]
        try:
            return cfg.labels.index(state_label)
        except ValueError:
            raise ValueError(f"Unknown state '{state_label}' for feature '{base_name}'")

    def encode_as_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert discretized labels to integer indices.

        Args:
            df: DataFrame with discretized features (label strings)

        Returns:
            DataFrame with integer-encoded features
        """
        result = df.copy()

        for feature_name in self.config.keys():
            col_name = f"{feature_name}_d"
            if col_name in df.columns:
                cfg = self.config[feature_name]
                # Create mapping from label to index
                label_to_idx = {label: idx for idx, label in enumerate(cfg.labels)}
                result[col_name] = df[col_name].map(label_to_idx)

        return result

    @classmethod
    def default(cls) -> 'ObservableDiscretizer':
        """Create discretizer with default configuration."""
        return cls(DEFAULT_CONFIGS.copy())

    def save_config(self, path: str) -> None:
        """Save fitted configuration to file."""
        import json

        config_dict = {
            "configs": {
                name: {
                    "bins": cfg.bins,
                    "labels": cfg.labels,
                    "use_percentiles": cfg.use_percentiles,
                }
                for name, cfg in self.config.items()
            },
            "fitted_bins": self._fitted_bins,
            "fitted": self._fitted,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> 'ObservableDiscretizer':
        """Load fitted configuration from file."""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = {
            name: DiscretizationConfig(**cfg)
            for name, cfg in config_dict["configs"].items()
        }

        discretizer = cls(config)
        discretizer._fitted_bins = config_dict["fitted_bins"]
        discretizer._fitted = config_dict["fitted"]

        return discretizer
