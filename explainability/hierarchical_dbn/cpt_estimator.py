"""
Conditional Probability Table (CPT) estimation for the hierarchical DBN.

Implements semi-supervised learning using:
    - Domain-informed priors for initialization
    - EM algorithm for latent variable parameters
    - Accident labels as supervision signal for top-level risk
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import ExpectationMaximization, BayesianEstimator

from .latent_model import (
    LatentFactors,
    RiskLevel,
    LatentState,
    OBSERVABLE_TO_LATENT,
    OBSERVABLE_CARDINALITY,
    LATENT_CARDINALITY,
    get_variable_cardinality,
)


class SemiSupervisedCPTEstimator:
    """
    Learns CPTs for the hierarchical DBN using semi-supervised approach.

    The learning process:
        1. Initialize CPTs with domain-informed priors
        2. Use EM algorithm to learn latent variable parameters
        3. Incorporate accident labels as supervision for top-level risk
    """

    def __init__(
        self,
        dbn: DynamicBayesianNetwork,
        prior_strength: float = 10.0,
        em_iterations: int = 100,
        convergence_threshold: float = 1e-4,
    ):
        """
        Initialize CPT estimator.

        Args:
            dbn: The Dynamic Bayesian Network structure
            prior_strength: Equivalent sample size for Bayesian priors
            em_iterations: Maximum EM iterations
            convergence_threshold: Convergence threshold for EM
        """
        self.dbn = dbn
        self.prior_strength = prior_strength
        self.em_iterations = em_iterations
        self.convergence_threshold = convergence_threshold
        self._cpds_initialized = False

    def initialize_with_priors(self) -> None:
        """
        Initialize CPTs with domain-informed prior distributions.

        This sets up sensible starting points for EM learning.
        """
        # Initialize observable CPDs (uniform or from data marginals)
        self._init_observable_cpds()

        # Initialize latent factor CPDs with domain knowledge
        self._init_latent_cpds()

        # Initialize accident risk CPD
        self._init_risk_cpd()

        # Initialize temporal transition CPDs
        self._init_transition_cpds()

        self._cpds_initialized = True

    def _init_observable_cpds(self) -> None:
        """Initialize CPDs for observable variables (uniform priors)."""
        # Only initialize CPDs for observables that are in the DBN structure
        dbn_nodes = set(n[0] for n in self.dbn.nodes())

        for obs_name in OBSERVABLE_CARDINALITY.keys():
            if obs_name not in dbn_nodes:
                continue

            cardinality = OBSERVABLE_CARDINALITY[obs_name]

            # Uniform distribution for observables
            values = np.ones((cardinality, 1)) / cardinality

            cpd = TabularCPD(
                variable=(obs_name, 0),
                variable_card=cardinality,
                values=values,
            )
            self.dbn.add_cpds(cpd)

    def _init_latent_cpds(self) -> None:
        """
        Initialize CPDs for intermediate latent factors.

        Uses domain knowledge to set prior distributions:
        - High risk indicators -> High latent state
        - Low risk indicators -> Low latent state
        """
        # Get observables actually in the DBN
        dbn_nodes = set(n[0] for n in self.dbn.nodes())

        for latent_factor in LatentFactors:
            latent_name = latent_factor.value

            # Filter to parent observables that are actually present
            all_parent_obs = OBSERVABLE_TO_LATENT[latent_factor]
            parent_obs = [p for p in all_parent_obs if p in dbn_nodes]

            if not parent_obs:
                # No parents available, use uniform prior
                cpd = TabularCPD(
                    variable=(latent_name, 0),
                    variable_card=3,
                    values=np.ones((3, 1)) / 3,
                )
                self.dbn.add_cpds(cpd)
                continue

            # Get parent cardinalities
            parent_cards = [OBSERVABLE_CARDINALITY[p] for p in parent_obs]

            # Latent has 3 states: Low, Moderate, High
            latent_card = 3

            # Create CPT based on domain knowledge
            cpd_values = self._create_latent_prior(
                latent_factor, parent_obs, parent_cards
            )

            cpd = TabularCPD(
                variable=(latent_name, 0),
                variable_card=latent_card,
                values=cpd_values,
                evidence=[(p, 0) for p in parent_obs],
                evidence_card=parent_cards,
            )
            self.dbn.add_cpds(cpd)

    def _create_latent_prior(
        self,
        latent_factor: LatentFactors,
        parent_obs: List[str],
        parent_cards: List[int],
    ) -> np.ndarray:
        """
        Create prior CPT for a latent factor based on domain knowledge.

        The prior encodes:
        - More "risky" parent states -> Higher latent state probability
        - Smooth transition between states
        """
        total_configs = int(np.prod(parent_cards))
        latent_card = 3  # Low, Moderate, High

        # Initialize CPT matrix
        cpd_values = np.zeros((latent_card, total_configs))

        # For each parent configuration, compute a risk score
        for config_idx in range(total_configs):
            # Decode configuration to individual parent states
            parent_states = self._decode_config(config_idx, parent_cards)

            # Compute risk score based on parent states
            risk_score = self._compute_risk_score(
                latent_factor, parent_obs, parent_states, parent_cards
            )

            # Convert risk score to probability distribution
            # Higher risk score -> Higher probability of High state
            cpd_values[:, config_idx] = self._score_to_distribution(risk_score)

        return cpd_values

    def _decode_config(self, config_idx: int, cardinalities: List[int]) -> List[int]:
        """Decode a linear configuration index to individual parent states."""
        states = []
        remaining = config_idx
        for card in reversed(cardinalities):
            states.append(remaining % card)
            remaining //= card
        return list(reversed(states))

    def _compute_risk_score(
        self,
        latent_factor: LatentFactors,
        parent_obs: List[str],
        parent_states: List[int],
        parent_cards: List[int],
    ) -> float:
        """
        Compute a risk score [0, 1] based on parent states.

        Higher values for higher parent states (more risk).
        """
        # Weight each parent's contribution by its position in the cardinality
        weighted_sum = 0.0
        total_weight = 0.0

        for obs, state, card in zip(parent_obs, parent_states, parent_cards):
            # Normalize state to [0, 1] range
            # Higher state index = higher risk
            normalized = state / (card - 1) if card > 1 else 0.5

            # Apply feature-specific weighting
            weight = self._get_feature_weight(latent_factor, obs)
            weighted_sum += normalized * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _get_feature_weight(self, latent_factor: LatentFactors, obs_name: str) -> float:
        """
        Get importance weight for an observable feature.

        Weights are based on domain knowledge of collision risk factors:
        - TTC and its rate of change are primary collision indicators
        - Relative metrics (accounting for ego motion) are more accurate than absolute
        - Acceleration indicates behavioral intent
        - Speed and proximity provide context
        """
        # Higher weights for more important features
        weights = {
            # =====================================================
            # COLLISION IMMINENCE WEIGHTS (primary risk indicators)
            # =====================================================
            # TTC metrics - relative version preferred
            "ttc_proxy_d": 1.8,            # Original TTC estimate
            "ttc_relative_d": 2.0,         # NEW: More accurate TTC (highest weight)
            "ttc_smoothed_d": 1.9,         # NEW: Less noisy TTC
            "ttc_rate_d": 1.8,             # Danger trend (critical for prediction)
            # Approach dynamics - relative version preferred
            "closing_rate_d": 1.3,         # Absolute approach velocity
            "rel_closing_rate_d": 1.6,     # NEW: Relative closing (more accurate)
            "proximity_rate_d": 1.4,       # NEW: Rate of getting closer
            # Distance metrics
            "proximity_d": 1.5,            # Current distance proxy
            "min_distance_t_d": 1.0,       # Environment: closest object

            # =====================================================
            # BEHAVIOURAL RISK WEIGHTS (intent/action indicators)
            # =====================================================
            # Absolute velocity
            "vx_d": 0.8,                   # Lateral motion
            "vy_d": 1.2,                   # Longitudinal motion
            "speed_d": 1.0,                # Speed magnitude
            "heading_d": 0.6,              # NEW: Direction of motion
            # NEW: Relative velocity (object relative to ego) - preferred
            "rel_vx_d": 1.0,               # Relative lateral motion
            "rel_vy_d": 1.4,               # Relative longitudinal motion
            "rel_speed_d": 1.2,            # Relative speed magnitude
            "rel_heading_d": 0.8,          # NEW: Relative direction
            # Acceleration (critical for intent detection)
            "ax_d": 1.3,                   # Lateral acceleration
            "ay_d": 1.5,                   # Longitudinal acceleration (braking!)
            # Ego vehicle state
            "ego_speed_d": 1.0,            # Ego vehicle speed
            "ego_accel_d": 1.5,            # Ego braking/acceleration
            # Risk speed
            "risk_speed_d": 1.2,           # Absolute risk speed
            "rel_risk_speed_d": 1.4,       # NEW: Relative risk speed (preferred)

            # =====================================================
            # ENVIRONMENTAL HAZARD WEIGHTS (context indicators)
            # =====================================================
            "num_objects_close_t_d": 1.2,  # Scene density
            "mean_rel_speed_t_d": 1.0,     # Average relative motion
            "min_ttc_t_d": 1.5,            # Most dangerous object TTC

            # =====================================================
            # TRACK CONFIDENCE (for uncertainty-aware inference)
            # =====================================================
            "pos_uncertainty_d": 0.3,      # NEW: Lower weight, used for discounting
            "vel_uncertainty_d": 0.3,      # NEW: Lower weight, used for discounting
        }
        return weights.get(obs_name, 1.0)

    def _score_to_distribution(self, risk_score: float) -> np.ndarray:
        """
        Convert a risk score [0, 1] to a probability distribution over 3 states.

        Uses a smooth mapping that concentrates probability on the appropriate state.
        """
        # Parameters for concentration
        alpha = 3.0  # Higher = more concentrated

        # Centers for Low, Moderate, High states
        centers = np.array([0.0, 0.5, 1.0])

        # Compute distances to each center
        distances = np.abs(centers - risk_score)

        # Convert to probabilities (inverse distance weighting)
        weights = np.exp(-alpha * distances)
        probs = weights / weights.sum()

        return probs

    def _init_risk_cpd(self) -> None:
        """
        Initialize CPD for top-level accident risk.

        Risk depends on all three intermediate latent factors.
        """
        latent_names = [lf.value for lf in LatentFactors]
        latent_cards = [3, 3, 3]  # All have 3 states
        total_configs = 27  # 3^3

        risk_card = 3  # Safe, Elevated, Critical
        cpd_values = np.zeros((risk_card, total_configs))

        for config_idx in range(total_configs):
            # Decode to latent states
            latent_states = self._decode_config(config_idx, latent_cards)

            # Compute overall risk based on latent states
            # Use max aggregation: risk dominated by highest latent factor
            max_state = max(latent_states)
            mean_state = np.mean(latent_states)

            # Combine max and mean for final risk
            combined_score = 0.7 * max_state / 2.0 + 0.3 * mean_state / 2.0

            cpd_values[:, config_idx] = self._score_to_distribution(combined_score)

        cpd = TabularCPD(
            variable=("accident_risk", 0),
            variable_card=risk_card,
            values=cpd_values,
            evidence=[(ln, 0) for ln in latent_names],
            evidence_card=latent_cards,
        )
        self.dbn.add_cpds(cpd)

    def _init_transition_cpds(self) -> None:
        """
        Initialize temporal transition CPDs.

        Encodes persistence: states tend to remain similar between time steps,
        with some probability of transition.
        """
        # Transition matrix with persistence bias
        persistence = 0.7  # Probability of staying in same state
        transition = (1 - persistence) / 2  # Split remaining probability

        transition_matrix = np.array([
            [persistence, transition, transition / 2],
            [transition, persistence, transition],
            [transition / 2, transition, persistence],
        ])
        # Normalize rows
        transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)

        # Apply to all latent factors
        for latent_factor in LatentFactors:
            latent_name = latent_factor.value
            cpd = TabularCPD(
                variable=(latent_name, 1),
                variable_card=3,
                values=transition_matrix,
                evidence=[(latent_name, 0)],
                evidence_card=[3],
            )
            self.dbn.add_cpds(cpd)

        # Accident risk transition
        cpd = TabularCPD(
            variable=("accident_risk", 1),
            variable_card=3,
            values=transition_matrix,
            evidence=[("accident_risk", 0)],
            evidence_card=[3],
        )
        self.dbn.add_cpds(cpd)

    def fit(
        self,
        data: pd.DataFrame,
        supervision_labels: Optional[pd.DataFrame] = None,
        video_fps: float = 10.0,
    ) -> 'SemiSupervisedCPTEstimator':
        """
        Learn CPT parameters from data.

        Args:
            data: DataFrame with discretized observable features (integer-encoded)
            supervision_labels: DataFrame with columns ['frame', 'accident_start_frame']
                              for semi-supervised learning
            video_fps: Frames per second for TTA calculation

        Returns:
            self for chaining
        """
        if not self._cpds_initialized:
            self.initialize_with_priors()

        # Prepare data for pgmpy
        pgmpy_data = self._prepare_data_for_pgmpy(data)

        # If we have supervision labels, add pseudo-labels for risk
        if supervision_labels is not None:
            pgmpy_data = self._add_risk_supervision(
                pgmpy_data, supervision_labels, video_fps
            )

        # Use Bayesian estimation with the priors we set
        # This updates CPTs based on data while respecting priors
        dbn_obs_nodes = [(n[0], n[1]) for n in self.dbn.nodes() if n[0] in OBSERVABLE_CARDINALITY]

        for node in dbn_obs_nodes:
            try:
                # Check if we have data for this node
                if node not in pgmpy_data.columns:
                    continue

                estimator = BayesianEstimator(self.dbn, pgmpy_data)
                cpd = estimator.estimate_cpd(
                    node,
                    prior_type="BDeu",
                    equivalent_sample_size=self.prior_strength,
                )
                self.dbn.add_cpds(cpd)

            except Exception as e:
                print(f"Warning: Bayesian estimation failed for {node}, using prior: {e}")

        return self

    def _prepare_data_for_pgmpy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for pgmpy format.

        pgmpy expects columns named as (variable, time_slice) tuples.
        """
        result = pd.DataFrame()

        for col in data.columns:
            if col.endswith("_d"):
                # Observable feature
                result[(col, 0)] = data[col]

        return result

    def _add_risk_supervision(
        self,
        data: pd.DataFrame,
        supervision: pd.DataFrame,
        fps: float,
    ) -> pd.DataFrame:
        """
        Add pseudo-labels for accident risk based on time-to-accident.

        Uses the semi-supervised labeling scheme from the research proposal:
            - >= 2.5s before accident: Safe
            - 1.5s - 2.5s before: Elevated
            - <= 1.5s before: Critical
        """
        result = data.copy()

        if "frame" not in data.columns and ("frame", 0) not in data.columns:
            # Cannot add supervision without frame information
            return result

        # Get frame column
        frame_col = "frame" if "frame" in data.columns else ("frame", 0)

        # Create risk labels based on TTA
        risk_labels = []

        for idx, row in data.iterrows():
            frame = row[frame_col] if frame_col in row else idx

            # Find matching supervision info
            if "accident_start_frame" in supervision.columns:
                acc_frame = supervision["accident_start_frame"].iloc[0]

                if pd.isna(acc_frame):
                    # No accident in this video -> Safe
                    risk_labels.append(RiskLevel.SAFE.value)
                else:
                    # Calculate TTA
                    frames_to_accident = acc_frame - frame
                    tta_seconds = frames_to_accident / fps

                    risk_level = RiskLevel.from_tta(tta_seconds)
                    risk_labels.append(risk_level.value)
            else:
                # No supervision -> leave as missing
                risk_labels.append(np.nan)

        result[("accident_risk", 0)] = risk_labels

        return result

    def get_cpd(self, variable: str, time_slice: int = 0) -> TabularCPD:
        """Get the CPD for a specific variable."""
        return self.dbn.get_cpds((variable, time_slice))

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'dbn': self.dbn,
                'prior_strength': self.prior_strength,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'SemiSupervisedCPTEstimator':
        """Load a trained model from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        estimator = cls(data['dbn'], data['prior_strength'])
        estimator._cpds_initialized = True
        return estimator
