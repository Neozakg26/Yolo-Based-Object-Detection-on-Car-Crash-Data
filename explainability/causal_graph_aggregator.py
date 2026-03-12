"""
Causal Graph Aggregation across multiple scenes.

Aggregates causal edges discovered via PCMCI from individual scenes into
a universal causal graph using p-value combination methods.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from networkx import DiGraph
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from explainability.feature_extractor import FeatureExtractor, FORBIDDEN_EDGES, EXPECTED_DIRECTIONS


class CausalGraphAggregator:
    """
    Aggregates causal graphs from multiple scenes into a universal graph.

    Supports multiple p-value combination methods:
    - Fisher's method: -2 * sum(log(p))
    - Stouffer's method: sum(z) / sqrt(n)
    - Vote counting: fraction of scenes with significant edge
    """

    def __init__(self, min_scenes: int = 2, method: str = "fisher"):
        """
        Initialize aggregator.

        Args:
            min_scenes: Minimum number of scenes an edge must appear in
            method: P-value combination method ('fisher', 'stouffer', 'vote')
        """
        self.min_scenes = min_scenes
        self.method = method

        # Storage for per-scene edge statistics
        self.scene_edges: Dict[str, pd.DataFrame] = {}  # scene_id -> edge_stats DataFrame
        self.scene_metadata: Dict[str, Dict] = {}  # scene_id -> metadata

        # Aggregated results
        self.universal_graph = DiGraph()
        self.edge_summary: Optional[pd.DataFrame] = None

    def add_scene(
        self,
        scene_id: str,
        edge_stats: pd.DataFrame,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add edge statistics from a single scene.

        Args:
            scene_id: Unique identifier for the scene
            edge_stats: DataFrame with columns: src, tgt, tau, count, p, weight, directions
            metadata: Optional scene metadata
        """
        self.scene_edges[scene_id] = edge_stats
        self.scene_metadata[scene_id] = metadata or {}
        print(f"Added scene {scene_id}: {len(edge_stats)} edge types")

    def add_scenes_from_directory(
        self,
        results_dir: str,
        tau_max: int = 2,
        pattern: str = "*_tracks.parquet"
    ) -> int:
        """
        Load and process scenes from a results directory.

        Args:
            results_dir: Directory containing parquet files
            tau_max: Maximum time lag for PCMCI
            pattern: Glob pattern for track files

        Returns:
            Number of scenes loaded
        """
        results_path = Path(results_dir)

        # Find track files
        track_files = list(results_path.glob(pattern))
        print(f"Found {len(track_files)} track files")

        n_loaded = 0
        for track_file in track_files:
            scene_id = track_file.stem.replace("_tracks", "")

            # Check for corresponding env file
            env_file = results_path / f"{scene_id}_env.parquet"
            if not env_file.exists():
                print(f"  Skipping {scene_id}: no environment file")
                continue

            # Check if edge_stats already exists
            edge_stats_file = results_path / f"{scene_id}_edge_stats.parquet"
            print(f"  loading Edge stats: {edge_stats_file}")
            if edge_stats_file.exists():
                # Load pre-computed edge stats
                edge_stats = pd.read_parquet(edge_stats_file)
                self.add_scene(scene_id, edge_stats)
                n_loaded += 1
            else:
                continue
                # # Run PCMCI to compute edge stats
                # print(f"  Computing PCMCI for {scene_id}...")
                # try:
                #     extractor = FeatureExtractor(
                #         track_path=str(track_file),
                #         env_path=str(env_file),
                #         tau_max=tau_max
                #     )
                #     edge_stats = extractor.extract_edges()
                #     self.add_scene(scene_id, edge_stats)
                #     n_loaded += 1
                # except Exception as e:
                #     print(f"  Error processing {scene_id}: {e}")
                #     continue

        print(f"Loaded {n_loaded} scenes")
        return n_loaded

    def aggregate(
        self,
        p_threshold: float = 0.01,
        min_weight: float = 0.10,
        min_direction_consistency: float = 0.6,
        apply_domain_constraints: bool = True
    ) -> DiGraph:
        """
        Aggregate edges across all scenes into a universal graph.

        Args:
            p_threshold: Maximum combined p-value for significance
            min_weight: Minimum average effect size
            min_direction_consistency: Minimum fraction of scenes agreeing on direction
            apply_domain_constraints: Filter forbidden edges

        Returns:
            NetworkX DiGraph with aggregated causal edges
        """
        print(f"\nAggregating {len(self.scene_edges)} scenes...")
        print(f"  Method: {self.method}")
        print(f"  Min scenes: {self.min_scenes}")
        print(f"  P-value threshold: {p_threshold}")
        print(f"  Min weight: {min_weight}")

        # Collect all edges across scenes
        all_edges: Dict[Tuple[str, str, int], Dict] = defaultdict(lambda: {
            "p": [],
            "weight": [],
            "directions": [],
            "scenes": [],
            "counts": []
        })

        for scene_id, edge_stats in self.scene_edges.items():
            print (f"scene ID: {scene_id}")
            print (f"edge stats : {edge_stats}")
            
            for _, row in edge_stats.iterrows():
                edge_key = (row["src"], row["tgt"], row["tau"])

                # Handle both list and scalar p
                p = row["p"] if isinstance(row["p"], list) else [row["p"]]
                weight = row["weight"] if isinstance(row["weight"], list) else [row["weight"]]
                directions = row["directions"] if isinstance(row["directions"], list) else [row["directions"]]

                all_edges[edge_key]["p"].extend(p)
                all_edges[edge_key]["weight"].extend(weight)
                all_edges[edge_key]["directions"].extend(directions)
                all_edges[edge_key]["scenes"].append(scene_id)
                all_edges[edge_key]["counts"].append(row.get("count", len(p)))

        print(f"  Total unique edges across scenes: {len(all_edges)}")

        # Build universal graph
        self.universal_graph = DiGraph()
        summary_rows = []

        filter_counts = {
            "insufficient_scenes": 0,
            "domain_constraint": 0,
            "insufficient_pvalue": 0,
            "insufficient_weight": 0,
            "inconsistent_direction": 0,
            "accepted": 0
        }

        for (src, tgt, tau), data in all_edges.items():
            # print(f"(src, tgt, tau): {src, tgt, tau}")
            # print(f"\n data: {data}")
            n_scenes = len(data["scenes"])

            # Filter 1: Minimum scene count
            if n_scenes < self.min_scenes:
                filter_counts["insufficient_scenes"] += 1
                continue

            # Filter 2: Domain constraints
            if apply_domain_constraints and (src, tgt) in FORBIDDEN_EDGES:
                filter_counts["domain_constraint"] += 1
                continue

            # Combine p-values
            p = np.array(data["p"], dtype=object)
            p_clipped = np.array(
                [np.clip(np.array(row,dtype=float), 1e-10, 1.0) for row in p]
                ,dtype=object )

            if self.method == "fisher":
                chi = -2 * sum(np.sum(np.log(np.array(row, dtype=float))) for row in p_clipped)
                p_combined = 1 - stats.chi2.cdf(chi, 2 * len(p))
            elif self.method == "stouffer":
                z_scores = stats.norm.ppf(1 - p_clipped)
                z_combined = np.sum(z_scores) / np.sqrt(len(z_scores))
                p_combined = 1 - stats.norm.cdf(z_combined)
            elif self.method == "vote":
                # Fraction of p-values below threshold
                significant = np.sum(p < p_threshold) / len(p)
                p_combined = 1 - significant if significant > 0.5 else 1.0
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Filter 3: Combined p-value
            if p_combined >= p_threshold:
                filter_counts["insufficient_pvalue"] += 1
                continue

            # Filter 4: Effect size
            weight = np.array(data["weight"], dtype=object)
            all_weight = np.concatenate([np.array(row, dtype=float) for row in weight])
            avg_weight = np.mean(all_weight)
            if avg_weight < min_weight:
                filter_counts["insufficient_weight"] += 1
                continue

            # Filter 5: Direction consistency
            directions = np.array(data["directions"], dtype= object)
            all_directiions = np.concatenate([np.array(row, dtype=float) for row in directions])
            pos_frac = np.mean( all_directiions > 0)
            neg_frac = np.mean(all_directiions < 0)
            direction_consistency = max(pos_frac, neg_frac)

            if direction_consistency < min_direction_consistency:
                filter_counts["inconsistent_direction"] += 1
                continue

            dominant_direction = "positive" if pos_frac > neg_frac else "negative"

            # Edge passes all filters
            filter_counts["accepted"] += 1

            # Add to universal graph
            self.universal_graph.add_edge(
                f"{src}(t-{tau})", f"{tgt}(t)",
                src=src,
                tgt=tgt,
                tau=tau,
                p_combined=p_combined,
                weight=avg_weight,
                weight_std=np.std(all_weight),
                n_scenes=n_scenes,
                scenes=data["scenes"],
                direction=dominant_direction,
                direction_consistency=direction_consistency,
                total_count=sum(data["counts"])
            )

            # Add to summary
            summary_rows.append({
                "src": src,
                "tgt": tgt,
                "tau": tau,
                "p_combined": p_combined,
                "weight_mean": avg_weight,
                "weight_std": np.std(all_weight),
                "n_scenes": n_scenes,
                "direction": dominant_direction,
                "direction_consistency": direction_consistency,
                "total_observations": sum(data["counts"])
            })

        # Store summary
        self.edge_summary = pd.DataFrame(summary_rows)
        if not self.edge_summary.empty:
            self.edge_summary = self.edge_summary.sort_values("weight_mean", ascending=False)

        # Print filter summary
        print(f"Edge filtering summary:")
        print(f"Insufficient scenes (<{self.min_scenes}): {filter_counts['insufficient_scenes']}")
        if apply_domain_constraints:
            print(f"  Domain constraint violations: {filter_counts['domain_constraint']}")
        print(f"  Insufficient p-value (>={p_threshold}): {filter_counts['insufficient_pvalue']}")
        print(f"  Insufficient weight (<{min_weight}): {filter_counts['insufficient_weight']}")
        print(f"  Inconsistent direction (<{min_direction_consistency}): {filter_counts['inconsistent_direction']}")
        print(f"  ACCEPTED: {filter_counts['accepted']}")

        return self.universal_graph

    def print_summary(self) -> None:
        """Print summary of aggregated graph."""
        print("\n" + "=" * 60)
        print("UNIVERSAL CAUSAL GRAPH SUMMARY")
        print("=" * 60)
        print(f"Total scenes: {len(self.scene_edges)}")
        print(f"Total nodes: {len(self.universal_graph.nodes())}")
        print(f"Total edges: {len(self.universal_graph.edges())}")
        print(f"Aggregation method: {self.method}")
        print(f"Min scenes threshold: {self.min_scenes}")

        if len(self.universal_graph.edges()) == 0:
            print("No significant edges in universal graph.")
            return

        # Group by tau
        tau_counts = defaultdict(int)
        for u, v, data in self.universal_graph.edges(data=True):
            tau_counts[data.get("tau", 1)] += 1

        print(f"Edges by time lag:")
        for tau in sorted(tau_counts.keys()):
            print(f"  tau={tau}: {tau_counts[tau]} edges")

        # Direction breakdown
        direction_counts = defaultdict(int)
        for u, v, data in self.universal_graph.edges(data=True):
            direction_counts[data.get("direction", "unknown")] += 1

        print(f"Edges by direction:")
        for direction, count in sorted(direction_counts.items()):
            print(f"  {direction}: {count} edges")

        # Top edges by weight
        print(f"Top 10 strongest edges:")
        edges = [
            (data.get("src"), data.get("tgt"), data.get("tau"), data.get("weight"), data.get("n_scenes"))
            for u, v, data in self.universal_graph.edges(data=True)
        ]
        edges.sort(key=lambda x: x[3], reverse=True)
        for src, tgt, tau, weight, n_scenes in edges[:10]:
            print(f"  {src} -> {tgt} (tau={tau}): weight={weight:.3f}, scenes={n_scenes}")

        print("=" * 60 + "\n")

    def validate_against_domain_knowledge(self) -> Dict[str, List]:
        """
        Validate universal graph against domain knowledge.

        Returns:
            Dictionary with 'consistent', 'inconsistent', 'unexpected' edges
        """
        validation = {
            "consistent": [],
            "inconsistent": [],
            "unexpected": []
        }

        for u, v, data in self.universal_graph.edges(data=True):
            src = data.get("src", u.split("(")[0])
            tgt = data.get("tgt", v.split("(")[0])
            direction = data.get("direction", "unknown")

            edge_pair = (src, tgt)

            if edge_pair in EXPECTED_DIRECTIONS:
                expected = EXPECTED_DIRECTIONS[edge_pair]
                if direction == expected:
                    validation["consistent"].append((src, tgt, direction))
                else:
                    validation["inconsistent"].append((src, tgt, direction, f"expected {expected}"))
            else:
                validation["unexpected"].append((src, tgt, direction))

        print(f"Domain knowledge validation:")
        print(f"Consistent with expectations: {len(validation['consistent'])}")
        print(f"Inconsistent with expectations: {len(validation['inconsistent'])}")
        print(f"Novel discoveries: {len(validation['unexpected'])}")

        if validation["inconsistent"]:
            print("Inconsistent edges (review these):")
            for edge in validation["inconsistent"]:
                print(f"    {edge[0]} -> {edge[1]}: {edge[2]} ({edge[3]})")

        return validation

    def get_edge_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of universal graph edges.

        Returns:
            DataFrame with edge statistics
        """
        if self.edge_summary is None or self.edge_summary.empty:
            return pd.DataFrame()
        return self.edge_summary.copy()

    def get_scene_agreement_matrix(self) -> pd.DataFrame:
        """
        Get matrix showing which scenes agree on which edges.

        Returns:
            DataFrame with edges as rows, scenes as columns
        """
        if len(self.universal_graph.edges()) == 0:
            return pd.DataFrame()

        # Get all scene IDs
        scene_ids = sorted(self.scene_edges.keys())

        # Build matrix
        rows = []
        for u, v, data in self.universal_graph.edges(data=True):
            edge_label = f"{data['src']}->{data['tgt']}(tau={data['tau']})"
            edge_scenes = set(data.get("scenes", []))

            row = {"edge": edge_label}
            for scene_id in scene_ids:
                row[scene_id] = 1 if scene_id in edge_scenes else 0
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("edge")
        return df

    def export_for_dbn(
        self,
        p_threshold: float = 0.01,
        min_weight: float = 0.10
    ) -> DiGraph:
        """
        Export universal graph for use in hierarchical DBN.

        Args:
            p_threshold: Maximum p-value threshold
            min_weight: Minimum weight threshold

        Returns:
            Filtered DiGraph suitable for DBN structure
        """
        dbn_graph = DiGraph()

        for u, v, data in self.universal_graph.edges(data=True):
            if data.get("p_combined", 1.0) >= p_threshold:
                continue
            if data.get("weight", 0) < min_weight:
                continue

            dbn_graph.add_edge(u, v, **data)

        print(f"Exported {len(dbn_graph.edges())} edges for DBN")
        return dbn_graph

    def draw_universal_graph(
        self,
        save_path: Optional[str] = None,
        show_scene_count: bool = True,
        figsize: Tuple[int, int] = (16, 12)
    ) -> None:
        """
        Visualize the universal causal graph.

        Args:
            save_path: Path to save figure
            show_scene_count: Show number of scenes in edge labels
            figsize: Figure size
        """
        if len(self.universal_graph.edges()) == 0:
            print("No edges to visualize")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Build positions (bipartite layout by tau)
        graph_nodes = set(self.universal_graph.nodes())
        tau_groups = defaultdict(list)
        target_nodes = []

        for node in graph_nodes:
            if "(t-" in node:
                tau_str = node.split("(t-")[1].rstrip(")")
                try:
                    tau = int(tau_str)
                    tau_groups[tau].append(node)
                except ValueError:
                    tau_groups[1].append(node)
            elif node.endswith("(t)"):
                target_nodes.append(node)

        pos = {}
        x_spacing = 4
        v_spacing = 1.2

        for tau in sorted(tau_groups.keys(), reverse=True):
            nodes = sorted(tau_groups[tau])
            x_pos = -tau * x_spacing
            for i, node in enumerate(nodes):
                pos[node] = (x_pos, -i * v_spacing)

        target_nodes = sorted(target_nodes)
        x_pos = x_spacing
        for i, node in enumerate(target_nodes):
            pos[node] = (x_pos, -i * v_spacing)

        # Edge styling
        edge_colors = []
        edge_widths = []
        edge_labels = {}

        for u, v, data in self.universal_graph.edges(data=True):
            direction = data.get("direction", "unknown")
            weight = data.get("weight", 0.1)
            n_scenes = data.get("n_scenes", 1)

            # Color by direction
            if direction == "positive":
                edge_colors.append("#2ca02c")
            elif direction == "negative":
                edge_colors.append("#d62728")
            else:
                edge_colors.append("#1f77b4")

            # Width by weight
            edge_widths.append(1.5 + 5 * weight)

            # Label
            label_parts = [f"w={weight:.2f}"]
            if show_scene_count:
                label_parts.append(f"n={n_scenes}")
            edge_labels[(u, v)] = "\n".join(label_parts)

        # Node colors by type
        node_colors = []
        for node in self.universal_graph.nodes():
            var_name = node.split("(")[0]
            if var_name in ["ttc_proxy", "ttc_rate", "min_ttc_t"]:
                node_colors.append("#FFB6C1")
            elif var_name in ["proximity", "closing_rate", "min_distance_t"]:
                node_colors.append("#98FB98")
            elif var_name in ["vx", "vy", "ax", "ay", "speed"]:
                node_colors.append("#ADD8E6")
            elif var_name in ["risk_speed", "mean_rel_speed_t"]:
                node_colors.append("#FFDAB9")
            else:
                node_colors.append("#D3D3D3")

        # Draw
        import networkx as nx
        nx.draw(
            self.universal_graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=2800,
            width=edge_widths,
            edge_color=edge_colors,
            font_size=8,
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1"
        )

        nx.draw_networkx_edge_labels(
            self.universal_graph, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="darkgreen",
            ax=ax
        )

        ax.set_title(
            f"Universal Causal Graph ({len(self.scene_edges)} scenes, {self.method} combination)",
            fontsize=14, fontweight="bold"
        )

        # Legend
        legend_elements = [
            Line2D([0], [0], color="#2ca02c", linewidth=3, label="Positive effect"),
            Line2D([0], [0], color="#d62728", linewidth=3, label="Negative effect"),
            Patch(facecolor="#FFB6C1", label="TTC-related"),
            Patch(facecolor="#98FB98", label="Proximity-related"),
            Patch(facecolor="#ADD8E6", label="Motion-related"),
            Patch(facecolor="#FFDAB9", label="Speed-risk"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Universal graph saved to {save_path}")

        plt.show()

    def save(self, path: str) -> None:
        """
        Save aggregator state to file.

        Args:
            path: Path to save (pickle format)
        """
        state = {
            "min_scenes": self.min_scenes,
            "method": self.method,
            "scene_edges": self.scene_edges,
            "scene_metadata": self.scene_metadata,
            "universal_graph": self.universal_graph,
            "edge_summary": self.edge_summary
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Aggregator saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CausalGraphAggregator":
        """
        Load aggregator from file.

        Args:
            path: Path to saved aggregator

        Returns:
            Loaded CausalGraphAggregator instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        aggregator = cls(
            min_scenes=state["min_scenes"],
            method=state["method"]
        )
        aggregator.scene_edges = state["scene_edges"]
        aggregator.scene_metadata = state["scene_metadata"]
        aggregator.universal_graph = state["universal_graph"]
        aggregator.edge_summary = state["edge_summary"]

        print(f"Loaded aggregator with {len(aggregator.scene_edges)} scenes")
        return aggregator
