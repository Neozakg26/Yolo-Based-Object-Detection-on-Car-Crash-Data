"""
High-level API for accident risk assessment.

Integrates discretization, DBN structure, CPT estimation, and inference
into a unified Class for end-to-end risk assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Iterable
from pathlib import Path
import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

from .discretizer import ObservableDiscretizer
from .dbn_structure import HierarchicalDBNStructure
from .cpt_estimator import SemiSupervisedCPTEstimator
from .dbn_inference import (
    BeliefPropagationInference,
    VariableEliminationInference)

from .discretizer import DiscretizationConfig


@dataclass
class RiskAssessment:
    """
    Container for risk assessment results.

    Provides both probabilistic risk estimates and explanations
    through intermediate latent factors.
    """
    # Probability distribution over risk states
    risk_distribution: Dict[str, float]

    # Distributions over intermediate latent factors
    latent_states: Dict[str, Dict[str, float]]

    # Most likely risk state (MAP estimate)
    most_likely_risk: str

    # Human-readable explanations
    contributing_factors: Dict[str, str]

    # Frame information
    frame_idx: Optional[int] = None

    @property
    def is_critical(self) -> bool:
        """Check if risk is predominantly critical."""
        return self.risk_distribution.get("CRITICAL", 0) > 0.5

    @property
    def is_elevated(self) -> bool:
        """Check if risk is elevated or higher."""
        return (
            self.risk_distribution.get("ELEVATED", 0) +
            self.risk_distribution.get("CRITICAL", 0)
        ) > 0.5

    @property
    def risk_score(self) -> float:
        """
        Scalar risk score [0, 1] for ranking/comparison.

        0 = Safe, 0.5 = Elevated, 1 = Critical
        """
        return (
            self.risk_distribution.get("SAFE", 0) * 0.0 +
            self.risk_distribution.get("ELEVATED", 0) * 0.5 +
            self.risk_distribution.get("CRITICAL", 0) * 1.0
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "frame_idx": self.frame_idx,
            "risk_distribution": self.risk_distribution,
            "most_likely_risk": self.most_likely_risk,
            "risk_score": self.risk_score,
            "latent_states": self.latent_states,
            "contributing_factors": self.contributing_factors,
        }

    def get_explanation(self) -> str:
        """Generate human-readable explanation of risk assessment."""
        lines = [f"Risk Level: {self.most_likely_risk} (score: {self.risk_score:.2f})"]
        lines.append("")
        lines.append("Contributing Factors:")

        for factor, explanation in self.contributing_factors.items():
            lines.append(f"  - {explanation}")

        lines.append("")
        lines.append("Risk Probabilities:")
        for level, prob in self.risk_distribution.items():
            lines.append(f"  - {level}: {prob:.1%}")

        return "\n".join(lines)


class AccidentRiskAssessor:
    """
    This Class exists For accident risk assessment.

    Integrates:
        - Feature discretization
        - Hierarchical DBN structure
        - CPT estimation
        - Real-time inference

    Usage:
        # Training
        assessor = AccidentRiskAssessor()
        assessor.fit(tracks_df, env_df, metadata)
        assessor.save("model.pkl")

        # Inference
        assessor = AccidentRiskAssessor.load("model.pkl")
        risk = assessor.assess_frame(frame_tracks, frame_env)
    """

    def __init__(
        self,
        inference_method: str = "supervised",
        prior_strength: float = 10.0,
        video_fps: float = 10.0,
        pretrained_classifier_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize risk assessor.

        Args:
            inference_method: "supervised", "belief_propagation", or "variable_elimination"
            prior_strength: Equivalent sample size for Bayesian priors
            video_fps: Frames per second for TTA calculation
            pretrained_classifier_path: Path to pre-trained classifier pickle file (optional).
                If provided, skips per-scene classifier training and uses the pre-trained model.
        """
        self.inference_method = inference_method
        self.prior_strength = prior_strength
        self.video_fps = video_fps

        self.discretizer: Optional[ObservableDiscretizer] = None
        self.dbn: Optional[DynamicBayesianNetwork] = None
        self.inference_engine: Optional[Union[
            BeliefPropagationInference,
            VariableEliminationInference
        ]] = None

        # Supervised classifier components
        self.classifier: Optional[GradientBoostingClassifier] = None
        self.ego_classifier: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []

        self._fitted = False
        self._pretrained_classifier_loaded = False

        # Load pre-trained classifier if path provided
        if pretrained_classifier_path is not None:
            self._load_pretrained_classifier(pretrained_classifier_path)

    def _load_pretrained_classifier(self, path: Union[str, Path]) -> None:
        """
        Load a pre-trained classifier from a pickle file.

        Args:
            path: Path to the pickle file containing classifier, scaler, and feature_cols
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pre-trained classifier not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        # Load risk classifier (support both old and new key names)
        self.classifier = model_data.get("risk_classifier", model_data.get("classifier"))
        self.scaler = model_data["scaler"]
        self.feature_cols = model_data["feature_cols"]

        # Load ego-involved classifier if available
        self.ego_classifier = model_data.get("ego_classifier", None)

        if "video_fps" in model_data:
            self.video_fps = model_data["video_fps"]

        self._pretrained_classifier_loaded = True

        n_scenes = model_data.get("n_scenes_trained", "unknown")
        n_frames = model_data.get("n_frames_trained", "unknown")
        has_ego = "with ego-involved" if self.ego_classifier is not None else "without ego-involved"
        print(f"Loaded pre-trained classifier ({has_ego}, trained on {n_scenes} scenes, {n_frames} frames)")

    @classmethod
    def with_pretrained_classifier(
        cls,
        classifier_path: Union[str, Path],
        inference_method: str = "supervised",
        prior_strength: float = 10.0,
    ) -> 'AccidentRiskAssessor':
        """
        Create an AccidentRiskAssessor with a pre-trained classifier.

        Args:
            classifier_path: Path to the pre-trained classifier pickle file
            inference_method: Inference method (default: "supervised")
            prior_strength: Prior strength for DBN CPT estimation

        Returns:
            AccidentRiskAssessor instance with pre-loaded classifier
        """
        return cls(
            inference_method=inference_method,
            prior_strength=prior_strength,
            pretrained_classifier_path=classifier_path,
        )

    def fit(
        self,
        tracks_df: pd.DataFrame,
        env_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict] = None,
        pcmci_graph=None,
    ) -> 'AccidentRiskAssessor':
        """
        Train the risk assessment model.

        Args:
            tracks_df: DataFrame with object tracking features
            env_df: DataFrame with environment features (optional)
            metadata: Dict with scene metadata including accident_start_frame
            pcmci_graph: NetworkX graph with PCMCI-discovered edges (optional)

        Returns:
            self for chaining
        """
        # Step 1: Merge environment features if provided
        if env_df is not None:
            tracks_df = tracks_df.merge(env_df, on="frame", how="left")

        # Step 2: Fit and apply discretization (only for features present in data)
        self.discretizer = ObservableDiscretizer.default()
        print(f"Runing Fit Transform: /n {tracks_df}")
        discrete_df = self.discretizer.fit_transform(tracks_df)

        # Step 3: Filter to observables actually present in discretized data
        available_obs = [col for col in discrete_df.columns if col.endswith("_d")]
        print(f"Available discretized features: {len(available_obs)}")

        # Step 4: Encode discrete labels as integers
        discrete_df = self.discretizer.encode_as_indices(discrete_df)

        # Store frame info
        discrete_df["frame"] = tracks_df["frame"].values

        # Step 5: Train supervised classifier if we have accident labels (skip if pre-trained)
        if self._pretrained_classifier_loaded:
            print("Using pre-trained classifier (skipping per-scene training)")
        elif metadata is not None and "accident_start_frame" in metadata:
            acc_frame = metadata["accident_start_frame"]
            if pd.notna(acc_frame):
                print(f"Training supervised classifier (accident at frame {int(acc_frame)})")
                self._train_supervised_classifier(discrete_df, int(acc_frame), available_obs)

        # Step 6: Build DBN structure (for compatibility)
        structure_builder = HierarchicalDBNStructure(
            observable_names=available_obs,
            pcmci_edges=pcmci_graph,
            include_pcmci_edges=(pcmci_graph is not None),
        )
        self.dbn = structure_builder.build()

        print(f"DBN structure: {len(self.dbn.nodes())} nodes, {len(self.dbn.edges())} edges")

        # Step 7: Estimate CPTs
        cpt_estimator = SemiSupervisedCPTEstimator(
            self.dbn,
            prior_strength=self.prior_strength,
        )

        supervision = None
        if metadata is not None and "accident_start_frame" in metadata:
            acc_frame = metadata["accident_start_frame"]
            if pd.notna(acc_frame):
                supervision = pd.DataFrame({
                    "accident_start_frame": [acc_frame]
                })

        cpt_estimator.fit(discrete_df, supervision_labels=supervision)

        # Step 8: Initialize inference engine (for non-supervised methods)
        if self.inference_method == "belief_propagation":
            self.inference_engine = BeliefPropagationInference(self.dbn)
        elif self.inference_method == "variable_elimination":
            self.inference_engine = VariableEliminationInference(self.dbn)

        self._fitted = True
        self._available_obs = available_obs
        return self

    def _train_supervised_classifier(
        self,
        discrete_df: pd.DataFrame,
        accident_frame: int,
        available_obs: List[str]
    ) -> None:
        """
        Train a supervised classifier using TTA-based labels.

        Args:
            discrete_df: DataFrame with encoded discrete features
            accident_frame: Frame number when accident occurs
            available_obs: List of available feature column names
        """
        # Create target labels based on Time-To-Accident
        discrete_df = discrete_df.copy()
        discrete_df["tta_frames"] = accident_frame - discrete_df["frame"]
        discrete_df["tta_seconds"] = discrete_df["tta_frames"] / self.video_fps

        # Assign risk labels
        def get_risk_label(tta):
            if tta >= 2.5:
                return 0  # SAFE
            elif tta >= 1.5:
                return 1  # ELEVATED
            else:
                return 2  # CRITICAL

        discrete_df["risk_label"] = discrete_df["tta_seconds"].apply(get_risk_label)

        # Aggregate features per frame (use most dangerous values)
        self.feature_cols = [col for col in available_obs if col in discrete_df.columns]

        # Aggregation: for each frame, get the riskiest feature values
        agg_dict = {col: "max" for col in self.feature_cols}
        agg_dict["risk_label"] = "first"  # Same for all rows in frame

        frame_features = discrete_df.groupby("frame").agg(agg_dict).reset_index()

        # Prepare features and labels
        X = frame_features[self.feature_cols].fillna(0).values
        y = frame_features["risk_label"].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier.fit(X_scaled, y)

        # Print training summary
        train_acc = self.classifier.score(X_scaled, y)
        print(f"  Classifier trained on {len(y)} frames")
        print(f"  Training accuracy: {train_acc:.1%}")
        print(f"  Label distribution: Safe={sum(y==0)}, Elevated={sum(y==1)}, Critical={sum(y==2)}")

    def assess_frame(
        self,
        track_features: pd.DataFrame,
        env_features: Optional[pd.Series] = None,
        frame_idx: Optional[int] = None,
    ) -> RiskAssessment:
        """
        Assess risk for a single video frame.

        Args:
            track_features: Object-level features for all tracks in frame
            env_features: Environment-level features (optional)
            frame_idx: Frame index for reference

        Returns:
            RiskAssessment with probabilities and explanations
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before assessment. Call fit() first.")

        # Step 1: Aggregate track features to frame level
        frame_obs = self._aggregate_to_frame(track_features, env_features)

        # Step 2: Discretize
        discrete_obs = self.discretizer.discretize_dict(frame_obs)

        # Step 3: Convert to integer indices
        int_obs = {
            name: self.discretizer.get_state_index(name, state)
            for name, state in discrete_obs.items()
        }

        # Step 4: Run inference
        self.inference_engine.update(int_obs)

        # Step 5: Format results
        risk_dist = self.inference_engine.get_risk_probability()
        latent_states = self.inference_engine.get_latent_states()
        map_risk = self.inference_engine.get_map_risk()

        # Generate explanations
        explanations = self._generate_explanations(latent_states, discrete_obs)

        return RiskAssessment(
            risk_distribution=risk_dist,
            latent_states=latent_states,
            most_likely_risk=map_risk,
            contributing_factors=explanations,
            frame_idx=frame_idx,
        )

    def assess_video(
        self,
        tracks_df: pd.DataFrame,
        env_df: Optional[pd.DataFrame] = None,
    ) -> List[RiskAssessment]:
        """
        Assess risk for an entire video sequence.

        Args:
            tracks_df: DataFrame with tracking features for all frames
            env_df: DataFrame with environment features (optional)

        Returns:
            List of RiskAssessment, one per frame
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before assessment.")

        # Merge environment features if provided
        if env_df is not None:
            merged_df = tracks_df.merge(env_df, on="frame", how="left")
        else:
            merged_df = tracks_df

        # Reset inference state for new video
        self.inference_engine.reset()

        assessments = []
        for frame_idx in sorted(merged_df["frame"].unique()):
            frame_tracks = merged_df[merged_df["frame"] == frame_idx]

            # Get environment features for this frame
            env_row = None
            if env_df is not None and frame_idx in env_df["frame"].values:
                env_row = env_df[env_df["frame"] == frame_idx].iloc[0]

            assessment = self.assess_frame(frame_tracks, env_row, frame_idx)
            assessments.append(assessment)

        return assessments

    def get_risk_trajectory(
        self,
        tracks_df: pd.DataFrame,
        env_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Get risk probability trajectory for a video.

        Uses supervised classifier if trained, otherwise falls back to direct scoring.

        Returns:
            DataFrame with frame index and risk probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before assessment.")

        # Merge environment features if provided
        if env_df is not None:
            merged_df = tracks_df.merge(env_df, on="frame", how="left")
        else:
            merged_df = tracks_df.copy()

        # Discretize and encode as integers
        discrete_df = self.discretizer.transform(merged_df)
        discrete_df = self.discretizer.encode_as_indices(discrete_df)
        discrete_df["frame"] = merged_df["frame"].values

        # Use supervised classifier if available
        if self.classifier is not None and self.inference_method == "supervised":
            return self._predict_with_classifier(discrete_df)
        else:
            return self._predict_with_direct_scoring(discrete_df, merged_df)

    def _predict_with_classifier(self, discrete_df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk and ego-involved using trained supervised classifiers."""
        records = []

        # Aggregate features per frame
        agg_dict = {col: "max" for col in self.feature_cols if col in discrete_df.columns}
        frame_features = discrete_df.groupby("frame").agg(agg_dict).reset_index()

        for _, row in frame_features.iterrows():
            frame_idx = row["frame"]

            # Prepare feature vector
            X = np.array([[row.get(col, 0) for col in self.feature_cols]])
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = self.scaler.transform(X)

            # Get risk class probabilities
            probs = self.classifier.predict_proba(X_scaled)[0]

            # Handle case where not all classes are present
            p_safe = probs[0] if len(probs) > 0 else 0.33
            p_elevated = probs[1] if len(probs) > 1 else 0.33
            p_critical = probs[2] if len(probs) > 2 else 0.34

            # Compute risk score (weighted average)
            risk_score = 0.0 * p_safe + 0.5 * p_elevated + 1.0 * p_critical

            # MAP estimate for risk
            risk_labels = ["Safe", "Elevated", "Critical"]
            map_risk = risk_labels[np.argmax([p_safe, p_elevated, p_critical])]

            record = {
                "frame": frame_idx,
                "P_Safe": p_safe,
                "P_Elevated": p_elevated,
                "P_Critical": p_critical,
                "risk_score": risk_score,
                "MAP_Risk": map_risk,
            }

            # Predict ego-involved if classifier is available
            if self.ego_classifier is not None:
                ego_probs = self.ego_classifier.predict_proba(X_scaled)[0]
                p_ego_no = ego_probs[0] if len(ego_probs) > 0 else 0.5
                p_ego_yes = ego_probs[1] if len(ego_probs) > 1 else 0.5
                map_ego = "Yes" if p_ego_yes > p_ego_no else "No"

                record["P_EgoInvolved_No"] = p_ego_no
                record["P_EgoInvolved_Yes"] = p_ego_yes
                record["MAP_EgoInvolved"] = map_ego

            records.append(record)

        return pd.DataFrame(records).sort_values("frame").reset_index(drop=True)

    def _predict_with_direct_scoring(
        self,
        discrete_df: pd.DataFrame,
        merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict risk using direct feature scoring (fallback method)."""
        records = []

        for frame_idx in sorted(merged_df["frame"].unique()):
            frame_data = discrete_df[discrete_df["frame"] == frame_idx]

            # Compute risk score directly from features
            risk_score = self._compute_direct_risk_score(frame_data)

            # Convert score to probabilities
            p_safe, p_elevated, p_critical = self._score_to_probs(risk_score)

            # Determine MAP risk
            probs = {"Safe": p_safe, "Elevated": p_elevated, "Critical": p_critical}
            map_risk = max(probs, key=probs.get)

            records.append({
                "frame": frame_idx,
                "P_Safe": p_safe,
                "P_Elevated": p_elevated,
                "P_Critical": p_critical,
                "risk_score": risk_score,
                "MAP_Risk": map_risk,
            })

        return pd.DataFrame(records)

    def _compute_direct_risk_score(self, frame_data: pd.DataFrame) -> float:
        """
        Compute risk score directly from discretized features (integer-encoded).

        Uses weighted combination of risk indicators.
        """
        from .latent_model import OBSERVABLE_CARDINALITY

        # Feature weights and whether to invert (weight, invert, default_cardinality)
        # invert=True means lower state index = higher risk (e.g., TTC: Critical=0)
        risk_weights = {
            # TTC features - lower TTC = higher risk (invert)
            "ttc_proxy_d": (2.0, True),
            "ttc_relative_d": (2.0, True),
            "ttc_smoothed_d": (2.0, True),
            "min_ttc_t_d": (1.5, True),
            "ttc_rate_d": (1.5, True),        # Rapid_Decrease=0 is dangerous
            # Proximity - higher index = closer = more risk (don't invert)
            "proximity_d": (1.5, False),
            "min_distance_t_d": (1.0, True),  # Danger=0 -> invert
            # Closing rate - higher = approaching faster (don't invert)
            "closing_rate_d": (1.3, False),
            "rel_closing_rate_d": (1.5, False),
            "proximity_rate_d": (1.2, False),
            # Speed - higher = faster = more risk
            "speed_d": (0.8, False),
            "rel_speed_d": (1.0, False),
            "risk_speed_d": (1.2, False),
            "rel_risk_speed_d": (1.3, False),
            # Velocity - approaching is higher risk
            "vy_d": (0.8, False),
            "rel_vy_d": (1.0, False),
        }

        total_score = 0.0
        total_weight = 0.0

        for col in frame_data.columns:
            if col not in risk_weights:
                continue

            weight, invert = risk_weights[col]

            # Get cardinality from latent model
            cardinality = OBSERVABLE_CARDINALITY.get(col, 3)
            max_state = cardinality - 1

            # Get values, convert to numeric
            values = pd.to_numeric(frame_data[col], errors='coerce').dropna()

            if len(values) == 0:
                continue

            if invert:
                # Lower state index = higher risk (e.g., TTC Critical=0)
                state_val = float(values.min())
                risk_val = 1.0 - (state_val / max_state) if max_state > 0 else 0.5
            else:
                # Higher state index = higher risk
                state_val = float(values.max())
                risk_val = state_val / max_state if max_state > 0 else 0.5

            # Clamp to [0, 1]
            risk_val = max(0.0, min(1.0, risk_val))

            total_score += weight * risk_val
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _score_to_probs(self, risk_score: float) -> Tuple[float, float, float]:
        """Convert risk score [0,1] to probability distribution over Safe/Elevated/Critical."""
        # Use softmax-like distribution centered on score
        alpha = 4.0  # Concentration parameter

        centers = np.array([0.2, 0.5, 0.8])  # Safe, Elevated, Critical centers
        distances = np.abs(centers - risk_score)
        weights = np.exp(-alpha * distances)
        probs = weights / weights.sum()

        return float(probs[0]), float(probs[1]), float(probs[2])

    def _aggregate_to_frame(
        self,
        track_features: pd.DataFrame,
        env_features: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Aggregate object-level features to frame-level observations.

        Uses risk-aware aggregation:
        - Minimum TTC (most dangerous object)
        - Maximum proximity (closest object)
        - Maximum closing rate
        """
        obs = {}

        # Object-level features: aggregate across tracks
        if "ttc_proxy" in track_features.columns:
            obs["ttc_proxy"] = track_features["ttc_proxy"].min()

        if "proximity" in track_features.columns:
            obs["proximity"] = track_features["proximity"].max()

        if "closing_rate" in track_features.columns:
            obs["closing_rate"] = track_features["closing_rate"].max()

        if "risk_speed" in track_features.columns:
            obs["risk_speed"] = track_features["risk_speed"].max()

        # Velocity: use most risky object (highest closing rate)
        if "closing_rate" in track_features.columns and len(track_features) > 0:
            risky_idx = track_features["closing_rate"].idxmax()
            if "vx" in track_features.columns:
                obs["vx"] = track_features.loc[risky_idx, "vx"]
            if "vy" in track_features.columns:
                obs["vy"] = track_features.loc[risky_idx, "vy"]

        # Ego motion: same for all tracks, take first
        if "ego_speed" in track_features.columns:
            obs["ego_speed"] = track_features["ego_speed"].iloc[0]

        if "ego_accel" in track_features.columns:
            obs["ego_accel"] = track_features["ego_accel"].iloc[0]

        # Environment features
        if env_features is not None:
            for col in ["min_distance_t", "mean_rel_speed_t", "min_ttc_t", "num_objects_close_t"]:
                if col in env_features.index:
                    obs[col] = env_features[col]

        return obs

    def _generate_explanations(
        self,
        latent_states: Dict[str, Dict[str, float]],
        discrete_obs: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate human-readable explanations for risk factors."""
        explanations = {}

        for latent_name, state_dist in latent_states.items():
            # Find most likely state
            max_state = max(state_dist, key=state_dist.get)
            confidence = state_dist[max_state]

            # Create explanation
            if latent_name == "collision_imminence":
                explanations["Collision Imminence"] = (
                    f"{max_state} ({confidence:.0%}) - "
                    f"TTC: {discrete_obs.get('ttc_proxy_d', 'N/A')}, "
                    f"Proximity: {discrete_obs.get('proximity_d', 'N/A')}"
                )
            elif latent_name == "behavioural_risk":
                explanations["Behavioural Risk"] = (
                    f"{max_state} ({confidence:.0%}) - "
                    f"Ego speed: {discrete_obs.get('ego_speed_d', 'N/A')}, "
                    f"Ego accel: {discrete_obs.get('ego_accel_d', 'N/A')}"
                )
            elif latent_name == "environmental_hazard":
                explanations["Environmental Hazard"] = (
                    f"{max_state} ({confidence:.0%}) - "
                    f"Objects nearby: {discrete_obs.get('num_objects_close_t_d', 'N/A')}"
                )

        return explanations

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the trained model to disk.
        And save the trained classifier to separate pickle file
        """
        path = Path(file_path)

        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model.")

        # Build DataFrame rows for model data
        rows = []

        # Add model parameters
        rows.append({
            "type": "param",
            "name": "inference_method",
            "value": self.inference_method,
            "bins": None,
            "labels": None,
            "use_percentiles": None,
            "fitted_bins": None,
        })
        rows.append({
            "type": "param",
            "name": "prior_strength",
            "value": str(self.prior_strength),
            "bins": None,
            "labels": None,
            "use_percentiles": None,
            "fitted_bins": None,
        })
        rows.append({
            "type": "param",
            "name": "video_fps",
            "value": str(self.video_fps),
            "bins": None,
            "labels": None,
            "use_percentiles": None,
            "fitted_bins": None,
        })

        # Add feature columns used by classifier
        rows.append({
            "type": "param",
            "name": "feature_cols",
            "value": ",".join(self.feature_cols) if self.feature_cols else "",
            "bins": None,
            "labels": None,
            "use_percentiles": None,
            "fitted_bins": None,
        })

        # Add discretizer config and fitted bins
        for name, cfg in self.discretizer.config.items():
            fitted_bins = self.discretizer._fitted_bins.get(name)
            rows.append({
                "type": "discretizer",
                "name": name,
                "value": None,
                "bins": cfg.bins,
                "labels": cfg.labels,
                "use_percentiles": cfg.use_percentiles,
                "fitted_bins": fitted_bins,
            })

        # Add DBN structure (edges)
        if self.dbn is not None:
            for u, v in self.dbn.edges():
                rows.append({
                    "type": "dbn_edge",
                    "name": f"{u[0]}_{u[1]}_to_{v[0]}_{v[1]}",
                    "value": f"{u}|{v}",
                    "bins": None,
                    "labels": None,
                    "use_percentiles": None,
                    "fitted_bins": None,
                })

        model_data = pd.DataFrame(rows)

        model_data.to_parquet(path, index=False, engine="pyarrow")

        # Save classifier separately as pickle (sklearn objects)
        if self.classifier is not None:
            classifier_path = path.with_suffix('.classifier.pkl')
            with open(classifier_path, 'wb') as f:                                                                                       
                pickle.dump({
                    'classifier': self.classifier,
                    'scaler': self.scaler,
                }, f)
            print(f"Classifier saved: {classifier_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AccidentRiskAssessor':

        path = Path(path)
        # if path.suffix == '.pkl':
        #     path = path.with_suffix('.parquet')

        model_data = pd.read_parquet(path, engine="pyarrow")
        print(f" Model \n {model_data}")

        # Extract parameters
        params = model_data[model_data["type"] == "param"]


        inference_method = params[params["name"] == "inference_method"]["value"].iloc[0]
        prior_strength = float(params[params["name"] == "prior_strength"]["value"].iloc[0])

        print(f"Extracted params: {params}")
        print(f"inf Method: {inference_method}")
        print(f"Prior Strength: {prior_strength}")

        # Get video_fps if available
        video_fps_row = params[params["name"] == "video_fps"]
        video_fps = float(video_fps_row["value"].iloc[0]) if len(video_fps_row) > 0 else 10.0

        # Get feature_cols if available
        feature_cols_row = params[params["name"] == "feature_cols"]
        feature_cols = []
        if len(feature_cols_row) > 0:
            fc_str = feature_cols_row["value"].iloc[0]
            if fc_str:
                feature_cols = fc_str.split(",")

        assessor = cls(
            inference_method=inference_method,
            prior_strength=prior_strength,
            video_fps=video_fps,
        )
        assessor.feature_cols = feature_cols

        # Restore discretizer config and fitted bins
        discretizer_rows = model_data[model_data["type"] == "discretizer"]
        config = {}
        fitted_bins = {}

        for _, row in discretizer_rows.iterrows():
            name = row["name"]
            config[name] = DiscretizationConfig(
                bins=row["bins"],
                labels=row["labels"],
                use_percentiles=row["use_percentiles"],
            )
            if row["fitted_bins"] is not None:
                fitted_bins[name] = row["fitted_bins"]

        assessor.discretizer = ObservableDiscretizer(config)
        assessor.discretizer._fitted_bins = fitted_bins
        assessor.discretizer._fitted = True

        # Rebuild DBN structure from edges
        dbn_edges = model_data[model_data["type"] == "dbn_edge"]
        if len(dbn_edges) > 0:

            assessor.dbn = DynamicBayesianNetwork()
            print(f"Inside loop:{len(dbn_edges)} for dbn_edges {dbn_edges}")
            for _, row in dbn_edges.iterrows():
                # Parse edge format: "(var, time)|(var, time)"
                edge_str = row["value"]
                print (f"ROW: {edge_str}")
                src_str, tgt_str = edge_str.split("|")
                # Convert string tuples back to actual tuples
                src = eval(src_str)
                tgt = eval(tgt_str)
                assessor.dbn.add_edge(src, tgt)

        # Load classifier if exists
        classifier_path = path.with_suffix('.classifier.pkl')
        print(f"Looking for Classifier with Path {classifier_path}")
        if classifier_path.exists():
            with open(classifier_path, 'rb') as f:
                clf_data = pickle.load(f)
                assessor.classifier = clf_data['classifier']
                assessor.scaler = clf_data['scaler']
            print(f"Classifier loaded: {classifier_path}")

        # Initialize inference engine (for non-supervised methods)
        if assessor.dbn is not None:
            if assessor.inference_method == "belief_propagation":
                assessor.inference_engine = BeliefPropagationInference(assessor.dbn)
            elif assessor.inference_method == "variable_elimination":
                assessor.inference_engine = VariableEliminationInference(assessor.dbn)

        assessor._fitted = True
        return assessor

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'AccidentRiskAssessor':
        """Alias for load() for convenience."""
        return cls.load(path)
    
    def fit_global(
        self,
        scene_items: Iterable[Tuple[str, pd.DataFrame, Optional[pd.DataFrame], Dict]],
        pcmci_graph=None,
        train_classifiers: bool = True,
        random_state: int = 42,
        frame_gap: int = 10_000,
    ) -> "AccidentRiskAssessor":
        """
        Global training across many scenes.

        Each item: (scene_id, tracks_df, env_df, metadata_dict)
        - tracks_df: object-level tracking rows with 'frame'
        - env_df: optional frame-level env rows with 'frame'
        - metadata_dict: contains 'accident_start_frame' (clip-based) and 'egoinvolve'

        Produces:
          - global discretizer
          - global DBN structure (optionally constrained by pcmci_graph)
          - global CPTs
          - optional global supervised classifiers (risk + ego)
        """
        # ---- 1) Merge env per scene + add scene_id + create frame_global to prevent cross-scene transitions ----
        merged_parts = []
        supervision_rows = []

        # We will keep both local and global frames.
        # local: original frame in scene (0..)
        # global: offset per scene with big gaps
        global_offset = 0

        for scene_id, tracks_df, env_df, meta in scene_items:
            df = tracks_df.copy()

            if env_df is not None:
                df = df.merge(env_df, on="frame", how="left")

            df["scene_id"] = scene_id
            df["frame_local"] = df["frame"].astype(int)

            # make a monotone global frame index with large gaps between scenes
            df["frame"] = df["frame_local"] + global_offset

            # choose next offset safely
            max_local = int(df["frame_local"].max()) if len(df) else 0
            global_offset += max(max_local + 1, 0) + int(frame_gap)

            merged_parts.append(df)

            # supervision row (per scene)
            acc = meta.get("accident_start_frame", None)
            # acc in meta is clip-based from CSV (1..50). we trained using preacc only, but store anyway:
            if acc is not None and pd.notna(acc):
                # convert to local 0-based for label computations if needed later
                acc_local = int(acc) - 1
            else:
                acc_local = None

            supervision_rows.append({
                "scene_id": scene_id,
                "accident_start_frame": acc_local,
                "egoinvolve": meta.get("egoinvolve", "Unknown"),
            })

        if not merged_parts:
            raise RuntimeError("fit_global(): no scenes provided.")

        merged_all = pd.concat(merged_parts, ignore_index=True)
        supervision_df = pd.DataFrame(supervision_rows)

        # ---- 2) Fit discretizer globally and encode ----
        self.discretizer = ObservableDiscretizer.default()
        discrete_df = self.discretizer.fit_transform(merged_all)

        available_obs = [col for col in discrete_df.columns if col.endswith("_d")]
        print(f"[GLOBAL] Available discretized features: {len(available_obs)}")

        discrete_df = self.discretizer.encode_as_indices(discrete_df)

        # Ensure essential columns exist
        discrete_df["frame"] = merged_all["frame"].values
        discrete_df["frame_local"] = merged_all["frame_local"].values
        discrete_df["scene_id"] = merged_all["scene_id"].values

        # ---- 3) Train global supervised classifiers (risk + ego) ----
        if train_classifiers:
            self._train_supervised_classifiers_global(
                discrete_df=discrete_df,
                supervision_df=supervision_df,
                available_obs=available_obs,
                random_state=random_state,
            )

        # ---- 4) Build DBN structure constrained by global PCMCI edges ----
        structure_builder = HierarchicalDBNStructure(
            observable_names=available_obs,
            pcmci_edges=pcmci_graph,
            include_pcmci_edges=(pcmci_graph is not None),
        )
        self.dbn = structure_builder.build()
        print(f"[GLOBAL] DBN structure: {len(self.dbn.nodes())} nodes, {len(self.dbn.edges())} edges")

        # ---- 5) Estimate CPTs globally ----
        cpt_estimator = SemiSupervisedCPTEstimator(
            self.dbn,
            prior_strength=self.prior_strength,
        )

        # Pass supervision if estimator uses it; keep only necessary columns
        # (scene_id may be ignored safely if estimator doesn't handle it)
        supervision_for_cpt = supervision_df[["accident_start_frame"]].copy()
        cpt_estimator.fit(discrete_df, supervision_labels=supervision_for_cpt)

        # ---- 6) Init inference engine if needed (for non-supervised inference) ----
        if self.inference_method == "belief_propagation":
            self.inference_engine = BeliefPropagationInference(self.dbn)
        elif self.inference_method == "variable_elimination":
            self.inference_engine = VariableEliminationInference(self.dbn)

        self._fitted = True
        self._available_obs = available_obs
        return self


    def _train_supervised_classifiers_global(
        self,
        discrete_df: pd.DataFrame,
        supervision_df: pd.DataFrame,
        available_obs: List[str],
        random_state: int = 42,
    ) -> None:
        """
        Train global classifiers:
          - risk classifier (Safe/Elevated/Critical) using TTA labels per scene
          - ego-involved classifier (Yes/No) using scene-level label replicated to frames

        IMPORTANT: uses scene-level grouping; no leakage across scenes.
        """
        # Feature columns available
        self.feature_cols = [col for col in available_obs if col in discrete_df.columns]
        if not self.feature_cols:
            print("[GLOBAL] No feature columns available for classifier training.")
            return

        # Build frame-level features per (scene_id, frame_local)
        # Use riskiest values per frame: max over tracks for each discretized feature.
        agg_dict = {col: "max" for col in self.feature_cols}
        frame_features = (
            discrete_df
            .groupby(["scene_id", "frame_local"], as_index=False)
            .agg(agg_dict)
        )

        # Attach supervision info
        frame_features = frame_features.merge(supervision_df, on="scene_id", how="left")

        # ---- Risk labels via Time-To-Accident ----
        # Only label scenes that have accident_start_frame
        labeled = frame_features[pd.notna(frame_features["accident_start_frame"])].copy()
        if labeled.empty:
            print("[GLOBAL] No scenes with accident_start_frame; skipping risk classifier training.")
        else:
            # Compute TTA in frames and seconds (local frames)
            labeled["tta_frames"] = labeled["accident_start_frame"].astype(int) - labeled["frame_local"].astype(int)
            labeled["tta_seconds"] = labeled["tta_frames"] / float(self.video_fps)

            def get_risk_label(tta_s: float) -> int:
                if tta_s >= 2.5:
                    return 0  # SAFE
                elif tta_s >= 1.5:
                    return 1  # ELEVATED
                else:
                    return 2  # CRITICAL

            labeled["risk_label"] = labeled["tta_seconds"].apply(get_risk_label)

            X = labeled[self.feature_cols].fillna(0).values
            y = labeled["risk_label"].values.astype(int)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.classifier.fit(X_scaled, y)

            train_acc = self.classifier.score(X_scaled, y)
            print(f"[GLOBAL] Risk classifier trained on {len(y)} frames across {labeled['scene_id'].nunique()} scenes")
            print(f"[GLOBAL] Risk training accuracy (in-sample): {train_acc:.1%}")
            print(f"[GLOBAL] Risk label distribution: Safe={sum(y==0)}, Elevated={sum(y==1)}, Critical={sum(y==2)}")

        # ---- Ego involvement classifier (scene-level label replicated) ----
        # Filter known labels
        ego_labeled = frame_features[frame_features["egoinvolve"].isin(["Yes", "No"])].copy()
        if ego_labeled.empty:
            print("[GLOBAL] No scenes with egoinvolve labels; skipping ego classifier training.")
            return

        y_ego = (ego_labeled["egoinvolve"] == "Yes").astype(int).values
        X_ego = ego_labeled[self.feature_cols].fillna(0).values

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_ego_scaled = self.scaler.fit_transform(X_ego)
        else:
            X_ego_scaled = self.scaler.transform(X_ego)

        self.ego_classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ego_classifier.fit(X_ego_scaled, y_ego)

        ego_acc = self.ego_classifier.score(X_ego_scaled, y_ego)
        print(f"[GLOBAL] Ego classifier trained on {len(y_ego)} frame-samples across {ego_labeled['scene_id'].nunique()} scenes")
        print(f"[GLOBAL] Ego training accuracy (in-sample): {ego_acc:.1%}")
        print(f"[GLOBAL] Ego label distribution: No={sum(y_ego==0)}, Yes={sum(y_ego==1)}")
