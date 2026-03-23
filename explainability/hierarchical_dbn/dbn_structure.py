"""
Dynamic Bayesian Network structure definition.

Builds the three-level hierarchical DBN with:
    - Intra-slice edges: observables -> latents -> risk
    - Inter-slice edges: temporal transitions + PCMCI-discovered edges
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from .latent_model import (
    LatentFactors,
    OBSERVABLE_TO_LATENT,
    ALL_OBSERVABLES,
    OBSERVABLE_CARDINALITY,
    LATENT_CARDINALITY,
    get_variable_cardinality,
)


class HierarchicalDBNStructure:
    """
    Builds the hierarchical Dynamic Bayesian Network structure.

    The DBN has three levels:
        1. Observable features (discretized tracking outputs)
        2. Intermediate latent factors (Collision Imminence, Behavioural Risk, Environmental Hazard)
        3. Top-level Accident Risk (Safe, Elevated, Critical)

    Supports hybrid structure learning:
        - Fixed hierarchical structure based on domain knowledge
        - PCMCI-discovered edges for temporal dependencies
    """

    def __init__(
        self,
        observable_names: Optional[List[str]] = None,
        pcmci_edges: Optional[nx.DiGraph] = None,
        include_pcmci_edges: bool = True,
    ):
        """
        Initialize DBN structure builder.

        Args:
            observable_names: List of observable feature names (with _d suffix).
                            If None, uses all observables from latent_model.
            pcmci_edges: NetworkX DiGraph with PCMCI-discovered causal edges.
                        Nodes should be in format "feature(t-1)" -> "feature(t)".
            include_pcmci_edges: Whether to incorporate PCMCI edges into structure.
        """
        self.observable_names = observable_names or ALL_OBSERVABLES.copy()
        self.pcmci_edges = pcmci_edges
        self.include_pcmci_edges = include_pcmci_edges
        self.dbn: Optional[DynamicBayesianNetwork] = None

    def build(self) -> DynamicBayesianNetwork:
        """
        Construct the hierarchical DBN.

        Returns:
            pgmpy DynamicBayesianNetwork with structure defined
        """
        self.dbn = DynamicBayesianNetwork()

        # Add intra-slice edges (within time slice 0)
        self._add_intra_slice_edges()

        # Add inter-slice edges (0 -> 1)
        self._add_inter_slice_edges()

        return self.dbn

    def _add_intra_slice_edges(self) -> None:
        """Add edges within a time slice (observables -> latents -> risk)."""

        # Layer 1 -> Layer 2: Observables -> Intermediate Latents
        for latent_factor, observables in OBSERVABLE_TO_LATENT.items():
            latent_name = latent_factor.value
            for obs in observables:
                if obs in self.observable_names:
                    # (observable, time=0) -> (latent, time=0)
                    self.dbn.add_edge((obs, 0), (latent_name, 0))

        # Layer 2 -> Layer 3: Intermediate Latents -> Accident Risk
        for latent_factor in LatentFactors:
            latent_name = latent_factor.value
            self.dbn.add_edge((latent_name, 0), ("accident_risk", 0))

    def _add_inter_slice_edges(self) -> None:
        """Add edges between time slices (temporal transitions)."""

        # Latent factor self-transitions (first-order Markov)
        for latent_factor in LatentFactors:
            latent_name = latent_factor.value
            self.dbn.add_edge((latent_name, 0), (latent_name, 1))

        # Top-level risk self-transition
        self.dbn.add_edge(("accident_risk", 0), ("accident_risk", 1))

        # Observable self-transitions (optional, for temporal smoothing)
        for obs in self.observable_names:
            self.dbn.add_edge((obs, 0), (obs, 1))

        # Add PCMCI-discovered inter-slice edges
        if self.include_pcmci_edges and self.pcmci_edges is not None:
            self._incorporate_pcmci_edges()

    def _incorporate_pcmci_edges(self) -> None:
        """
        Add PCMCI-discovered edges to inter-slice connections.

        PCMCI nodes are in format: "feature(t-1)" -> "feature(t)"
        We parse these and add corresponding edges to the DBN.
        """
        for u, v, data in self.pcmci_edges.edges(data=True):
            # Parse node names like "ttc_proxy(t-1)" and "ttc_proxy(t)"
            src_var, src_time = self._parse_pcmci_node(u)
            tgt_var, tgt_time = self._parse_pcmci_node(v)

            if src_var is None or tgt_var is None:
                continue

            # Map to discretized variable names
            src_discrete = f"{src_var}_d" if not src_var.endswith("_d") else src_var
            tgt_discrete = f"{tgt_var}_d" if not tgt_var.endswith("_d") else tgt_var

            # Only add if both variables are in our model
            if src_discrete not in self.observable_names:
                continue
            if tgt_discrete not in self.observable_names:
                continue

            # Add inter-slice edge (time 0 -> time 1)
            # PCMCI edges represent t-lag -> t, which maps to 0 -> 1 in DBN
            try:
                self.dbn.add_edge((src_discrete, 0), (tgt_discrete, 1))
            except Exception:
                # Edge may already exist
                pass

    def _parse_pcmci_node(self, node_str: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a PCMCI node string like "ttc_proxy(t-1)" or "ttc_proxy(t)".

        Returns:
            Tuple of (variable_name, time_offset) or (None, None) if parsing fails
        """
        if "(" not in node_str or ")" not in node_str:
            return None, None

        var_name = node_str.split("(")[0]
        time_str = node_str.split("(")[1].rstrip(")")

        if time_str == "t":
            return var_name, 0
        elif time_str.startswith("t-"):
            try:
                offset = int(time_str[2:])
                return var_name, -offset
            except ValueError:
                return None, None

        return None, None

    def get_structure_info(self) -> Dict:
        """
        Get information about the DBN structure.

        Returns:
            Dict with node counts, edge counts, and structure details
        """
        if self.dbn is None:
            raise RuntimeError("DBN not built. Call build() first.")

        intra_edges = [
            (u, v) for u, v in self.dbn.edges()
            if u[1] == v[1]  # Same time slice
        ]
        inter_edges = [
            (u, v) for u, v in self.dbn.edges()
            if u[1] != v[1]  # Different time slices
        ]

        return {
            "num_nodes_per_slice": len(set(n[0] for n in self.dbn.nodes())),
            "num_intra_edges": len(intra_edges),
            "num_inter_edges": len(inter_edges),
            "observables": self.observable_names,
            "latent_factors": [lf.value for lf in LatentFactors],
            "intra_edges": intra_edges,
            "inter_edges": inter_edges,
        }

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the DBN structure.

        Args:
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt

        if self.dbn is None:
            raise RuntimeError("DBN not built. Call build() first.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        # Time slice 0
        self._draw_time_slice(axes[0], time=0, title="Time Slice t")

        # Time slice 1 with inter-slice edges
        self._draw_time_slice(axes[1], time=1, title="Time Slice t+1")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def _draw_time_slice(self, ax, time: int, title: str) -> None:
        """Draw a single time slice of the DBN."""
        import matplotlib.patches as mpatches

        # Get nodes for this time slice
        nodes_t = [n for n in self.dbn.nodes() if n[1] == time]

        # Separate by layer
        observables = [n for n in nodes_t if n[0] in self.observable_names]
        latents = [n for n in nodes_t if n[0] in [lf.value for lf in LatentFactors]]
        risk = [n for n in nodes_t if n[0] == "accident_risk"]

        # Position nodes in layers
        pos = {}

        # Observables (bottom layer)
        for i, node in enumerate(sorted(observables)):
            pos[node] = (i * 1.5, 0)

        # Latents (middle layer)
        latent_x_start = (len(observables) * 1.5 - len(latents) * 2) / 2
        for i, node in enumerate(sorted(latents)):
            pos[node] = (latent_x_start + i * 2, 2)

        # Risk (top layer)
        for node in risk:
            pos[node] = (len(observables) * 1.5 / 2 - 0.75, 4)

        # Create subgraph for this time slice
        subgraph = self.dbn.subgraph(nodes_t)

        # Draw
        nx.draw(
            subgraph,
            pos,
            ax=ax,
            with_labels=True,
            labels={n: n[0][:12] for n in nodes_t},  # Truncate long names
            node_color=[
                "#90EE90" if n in observables else
                "#FFD700" if n in latents else
                "#FF6B6B"
                for n in nodes_t
            ],
            node_size=1500,
            font_size=7,
            font_weight="bold",
            arrows=True,
            arrowsize=15,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")

        # Legend
        legend_elements = [
            mpatches.Patch(color="#90EE90", label="Observables"),
            mpatches.Patch(color="#FFD700", label="Latent Factors"),
            mpatches.Patch(color="#FF6B6B", label="Accident Risk"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")


def get_fixed_structure() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get the fixed hierarchical structure definition.

    Returns:
        Dict with 'intra_slice' and 'inter_slice' edge lists
    """
    intra_edges = []
    inter_edges = []

    # Observables -> Latents
    for latent_factor, observables in OBSERVABLE_TO_LATENT.items():
        for obs in observables:
            intra_edges.append((obs, latent_factor.value))

    # Latents -> Risk
    for latent_factor in LatentFactors:
        intra_edges.append((latent_factor.value, "accident_risk"))

    # Temporal transitions for latents and risk
    for latent_factor in LatentFactors:
        inter_edges.append((latent_factor.value, latent_factor.value))
    inter_edges.append(("accident_risk", "accident_risk"))

    return {
        "intra_slice": intra_edges,
        "inter_slice": inter_edges,
    }
