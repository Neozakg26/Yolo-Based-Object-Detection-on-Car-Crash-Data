"""
Inference engines for the hierarchical DBN.

Implements both Belief Propagation (fast, approximate) and
Variable Elimination (exact, slower) for risk state estimation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork, BayesianNetwork
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor

from .latent_model import (
    LatentFactors,
    RiskLevel,
    OBSERVABLE_TO_LATENT,
    ALL_OBSERVABLES,
)


class DBNInferenceEngine(ABC):
    """Abstract base class for DBN inference engines."""

    def __init__(self, dbn: DynamicBayesianNetwork):
        """
        Initialize inference engine.

        Args:
            dbn: The Dynamic Bayesian Network with CPDs defined
        """
        self.dbn = dbn
        self.belief_state: Optional[Dict[str, np.ndarray]] = None
        self._time_step = 0

    @abstractmethod
    def initialize(self, initial_observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Initialize belief state from first observation.

        Args:
            initial_observations: Dict mapping observable names to discrete state indices

        Returns:
            Dict with posterior distributions over all latent variables
        """
        pass

    @abstractmethod
    def update(self, observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Perform one-step belief update given new observations.

        Args:
            observations: Dict mapping observable names to discrete state indices

        Returns:
            Dict with posterior distributions over all latent variables
        """
        pass

    def get_risk_probability(self) -> Dict[str, float]:
        """
        Get current probability distribution over accident risk.

        Returns:
            Dict mapping risk levels to probabilities
        """
        if self.belief_state is None:
            raise RuntimeError("Must call update() first")

        risk_dist = self.belief_state.get("accident_risk", np.array([1/3, 1/3, 1/3]))

        return {
            RiskLevel.SAFE.name: float(risk_dist[0]),
            RiskLevel.ELEVATED.name: float(risk_dist[1]),
            RiskLevel.CRITICAL.name: float(risk_dist[2]),
        }

    def get_latent_states(self) -> Dict[str, Dict[str, float]]:
        """
        Get probability distributions over intermediate latent factors.

        Returns:
            Dict mapping latent factor names to state distributions
        """
        if self.belief_state is None:
            raise RuntimeError("Must call update() first")

        result = {}
        for latent_factor in LatentFactors:
            latent_name = latent_factor.value
            dist = self.belief_state.get(latent_name, np.array([1/3, 1/3, 1/3]))
            result[latent_name] = {
                "Low": float(dist[0]),
                "Moderate": float(dist[1]),
                "High": float(dist[2]),
            }

        return result

    def get_map_risk(self) -> str:
        """Get the most likely risk state."""
        risk_probs = self.get_risk_probability()
        return max(risk_probs, key=risk_probs.get)

    def reset(self) -> None:
        """Reset the inference state for a new video."""
        self.belief_state = None
        self._time_step = 0


class BeliefPropagationInference(DBNInferenceEngine):
    """
    Loopy Belief Propagation inference for fast, approximate inference.

    Suitable for real-time applications where speed is critical.
    May not converge for graphs with many loops.
    """

    def __init__(
        self,
        dbn: DynamicBayesianNetwork,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ):
        """
        Initialize Belief Propagation inference.

        Args:
            dbn: The Dynamic Bayesian Network
            max_iterations: Maximum BP iterations
            convergence_threshold: Convergence threshold for message passing
        """
        super().__init__(dbn)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._bn_t0: Optional[BayesianNetwork] = None
        self._bp_engine: Optional[BeliefPropagation] = None

    def _get_time_slice_bn(self, time_slice: int = 0) -> BayesianNetwork:
        """
        Extract a single time slice as a regular Bayesian Network.

        This allows us to use standard BP inference on each slice.
        """
        bn = BayesianNetwork()

        # Add nodes and edges for this time slice
        for node in self.dbn.nodes():
            if node[1] == time_slice:
                bn.add_node(node[0])

        for u, v in self.dbn.edges():
            if u[1] == time_slice and v[1] == time_slice:
                bn.add_edge(u[0], v[0])

        # Add CPDs (strip time index from variable names)
        for cpd in self.dbn.get_cpds():
            var_name, var_time = cpd.variable
            if var_time == time_slice:
                # Check if this is an intra-slice CPD
                evidence = cpd.get_evidence() if hasattr(cpd, 'get_evidence') else []
                intra_slice_evidence = [e for e in evidence if e[1] == time_slice]

                if len(evidence) == 0 or len(intra_slice_evidence) == len(evidence):
                    # Create new CPD without time indices
                    new_evidence = [e[0] for e in intra_slice_evidence]
                    evidence_card = cpd.evidence_card if len(new_evidence) > 0 else None

                    new_cpd = TabularCPD(
                        variable=var_name,
                        variable_card=cpd.variable_card,
                        values=cpd.values,
                        evidence=new_evidence if new_evidence else None,
                        evidence_card=list(evidence_card) if evidence_card is not None else None,
                    )
                    bn.add_cpds(new_cpd)

        return bn

    def initialize(self, initial_observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Initialize belief state from first observation."""
        # Get time slice 0 as regular BN
        self._bn_t0 = self._get_time_slice_bn(0)

        # Create BP engine
        self._bp_engine = BeliefPropagation(self._bn_t0)

        # Perform inference
        return self.update(initial_observations)

    def update(self, observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Perform belief propagation inference."""
        if self._bp_engine is None:
            return self.initialize(observations)

        # Format evidence for pgmpy (strip _d suffix if needed, use base name)
        evidence = {}
        for obs_name, state_idx in observations.items():
            # Use the observable name without suffix for the BN
            base_name = obs_name
            if base_name in [n for n in self._bn_t0.nodes()]:
                evidence[base_name] = state_idx

        # Query latent variables and risk
        query_vars = [lf.value for lf in LatentFactors] + ["accident_risk"]
        query_vars = [v for v in query_vars if v in self._bn_t0.nodes()]

        try:
            # Perform BP inference
            result = self._bp_engine.query(
                variables=query_vars,
                evidence=evidence,
                show_progress=False,
            )

            # Extract distributions
            self.belief_state = {}
            for var in query_vars:
                if var in result.variables:
                    self.belief_state[var] = result.values
                elif hasattr(result, 'get_value'):
                    # Handle joint distribution case
                    factor = self._bp_engine.query([var], evidence=evidence)
                    self.belief_state[var] = factor.values

        except Exception as e:
            # Fallback to uniform distribution on error
            print(f"BP inference warning: {e}")
            self.belief_state = {
                var: np.array([1/3, 1/3, 1/3])
                for var in query_vars
            }

        self._time_step += 1
        return self.belief_state


class VariableEliminationInference(DBNInferenceEngine):
    """
    Variable Elimination inference for exact inference.

    Slower than BP but guarantees correct results.
    Better for validation and when accuracy is critical.
    """

    def __init__(self, dbn: DynamicBayesianNetwork):
        """
        Initialize Variable Elimination inference.

        Args:
            dbn: The Dynamic Bayesian Network
        """
        super().__init__(dbn)
        self._bn_t0: Optional[BayesianNetwork] = None
        self._ve_engine: Optional[VariableElimination] = None

    def _get_time_slice_bn(self, time_slice: int = 0) -> BayesianNetwork:
        """Extract a single time slice as a regular Bayesian Network."""
        bn = BayesianNetwork()

        # Add nodes for this time slice
        for node in self.dbn.nodes():
            if node[1] == time_slice:
                bn.add_node(node[0])

        # Add intra-slice edges
        for u, v in self.dbn.edges():
            if u[1] == time_slice and v[1] == time_slice:
                bn.add_edge(u[0], v[0])

        # Add CPDs
        for cpd in self.dbn.get_cpds():
            var_name, var_time = cpd.variable
            if var_time == time_slice:
                evidence = cpd.get_evidence() if hasattr(cpd, 'get_evidence') else []
                intra_evidence = [e for e in evidence if e[1] == time_slice]

                if len(evidence) == 0 or len(intra_evidence) == len(evidence):
                    new_evidence = [e[0] for e in intra_evidence]
                    evidence_card = cpd.evidence_card if len(new_evidence) > 0 else None

                    new_cpd = TabularCPD(
                        variable=var_name,
                        variable_card=cpd.variable_card,
                        values=cpd.values,
                        evidence=new_evidence if new_evidence else None,
                        evidence_card=list(evidence_card) if evidence_card is not None else None,
                    )
                    bn.add_cpds(new_cpd)

        return bn

    def initialize(self, initial_observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Initialize with Variable Elimination."""
        self._bn_t0 = self._get_time_slice_bn(0)
        self._ve_engine = VariableElimination(self._bn_t0)
        return self.update(initial_observations)

    def update(self, observations: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Perform Variable Elimination inference."""
        if self._ve_engine is None:
            return self.initialize(observations)

        # Format evidence
        evidence = {}
        for obs_name, state_idx in observations.items():
            if obs_name in [n for n in self._bn_t0.nodes()]:
                evidence[obs_name] = state_idx

        # Query each latent variable separately for VE
        query_vars = [lf.value for lf in LatentFactors] + ["accident_risk"]
        query_vars = [v for v in query_vars if v in self._bn_t0.nodes()]

        self.belief_state = {}

        for var in query_vars:
            try:
                result = self._ve_engine.query(
                    variables=[var],
                    evidence=evidence,
                    show_progress=False,
                )
                self.belief_state[var] = result.values

            except Exception as e:
                print(f"VE inference warning for {var}: {e}")
                self.belief_state[var] = np.array([1/3, 1/3, 1/3])

        self._time_step += 1
        return self.belief_state


class SequentialDBNInference:
    """
    Sequential inference over a video sequence using filtering.

    Maintains belief state across frames, incorporating temporal transitions.
    """

    def __init__(
        self,
        dbn: DynamicBayesianNetwork,
        inference_method: str = "belief_propagation",
    ):
        """
        Initialize sequential inference.

        Args:
            dbn: The Dynamic Bayesian Network
            inference_method: "belief_propagation" or "variable_elimination"
        """
        self.dbn = dbn

        if inference_method == "belief_propagation":
            self.engine = BeliefPropagationInference(dbn)
        elif inference_method == "variable_elimination":
            self.engine = VariableEliminationInference(dbn)
        else:
            raise ValueError(f"Unknown inference method: {inference_method}")

        self._prev_belief: Optional[Dict[str, np.ndarray]] = None

    def filter_sequence(
        self,
        observations_sequence: List[Dict[str, int]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform filtering over a sequence of observations.

        Args:
            observations_sequence: List of observation dicts, one per time step

        Returns:
            List of belief state dicts, one per time step
        """
        beliefs = []

        self.engine.reset()
        self._prev_belief = None

        for t, observations in enumerate(observations_sequence):
            # Incorporate temporal transition from previous belief
            if self._prev_belief is not None:
                observations = self._incorporate_temporal_prior(observations)

            # Update belief
            belief = self.engine.update(observations)
            beliefs.append(belief.copy())

            self._prev_belief = belief

        return beliefs

    def _incorporate_temporal_prior(
        self,
        observations: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Incorporate temporal transition probabilities.

        This is a simplified approach that uses the previous belief
        as a soft prior for the current inference.
        """
        # For now, just return observations unchanged
        # More sophisticated approaches would modify the CPDs or evidence
        return observations

    def get_risk_trajectory(
        self,
        observations_sequence: List[Dict[str, int]],
    ) -> pd.DataFrame:
        """
        Get risk probability trajectory over the sequence.

        Returns:
            DataFrame with columns for each risk level probability
        """
        beliefs = self.filter_sequence(observations_sequence)

        records = []
        for t, belief in enumerate(beliefs):
            risk_dist = belief.get("accident_risk", np.array([1/3, 1/3, 1/3]))
            records.append({
                "frame": t,
                "P_Safe": risk_dist[0],
                "P_Elevated": risk_dist[1],
                "P_Critical": risk_dist[2],
                "MAP_Risk": ["Safe", "Elevated", "Critical"][np.argmax(risk_dist)],
            })

        return pd.DataFrame(records)
