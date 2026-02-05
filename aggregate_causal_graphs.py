"""
Script to aggregate causal graphs from multiple scenes into a universal graph.

Usage:
    python aggregate_causal_graphs.py --results_dir path/to/results
    python aggregate_causal_graphs.py --results_dir path/to/results --min_scenes 3 --method fisher
"""

import argparse
from pathlib import Path
from explainability.causal_graph_aggregator import CausalGraphAggregator


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate causal graphs from multiple scenes"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing tracking results (*_tracks.parquet, *_env.parquet)")
    
    parser.add_argument(
        "--min_scenes",
        type=int,
        default=2,
        help="Minimum number of scenes an edge must appear in (default: 2)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fisher", "stouffer", "vote"],
        default="fisher",
        help="P-value combination method (default: fisher)"
    )
    parser.add_argument(
        "--tau_max",
        type=int,
        default=2,
        help="Maximum time lag for PCMCI (default: 2)"
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.01,
        help="P-value threshold for edge inclusion (default: 0.01)"
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.10,
        help="Minimum effect size for edge inclusion (default: 0.10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: results_dir)"
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip visualization"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize aggregator
    print(f"\n{'='*60}")
    print("CAUSAL GRAPH AGGREGATION PIPELINE")
    print(f"{'='*60}")

    aggregator = CausalGraphAggregator(
        min_scenes=args.min_scenes,
        method=args.method
    )

    # Load all scenes from directory
    print(f"Loading scenes from: {args.results_dir}")
    n_scenes = aggregator.add_scenes_from_directory(
        results_dir=args.results_dir,
        tau_max=args.tau_max
    )

    if n_scenes < args.min_scenes:
        print(f"Error: Found only {n_scenes} scenes, need at least {args.min_scenes}")
        return

    # Aggregate graphs
    universal_graph = aggregator.aggregate(
        p_threshold=args.p_threshold,
        min_weight=args.min_weight,
        min_direction_consistency=0.6,
        apply_domain_constraints=True
    )

    # Print summary
    aggregator.print_summary()

    # Validate against domain knowledge
    validation = aggregator.validate_against_domain_knowledge()

    # Save results
    # 1. Save aggregator state
    aggregator_path = output_dir / "universal_causal_aggregator.pkl"
    aggregator.save(str(aggregator_path))

    # 2. Save edge summary as CSV
    edge_summary = aggregator.get_edge_summary()
    if not edge_summary.empty:
        summary_path = output_dir / "universal_causal_edges.csv"
        edge_summary.to_csv(summary_path, index=False)
        print(f"Edge summary saved to: {summary_path}")

    # 3. Save scene agreement matrix
    agreement_matrix = aggregator.get_scene_agreement_matrix()
    if not agreement_matrix.empty:
        matrix_path = output_dir / "scene_edge_agreement.csv"
        agreement_matrix.to_csv(matrix_path)
        print(f"Scene agreement matrix saved to: {matrix_path}")

    # 4. Export for DBN
    dbn_graph = aggregator.export_for_dbn(
        p_threshold=args.p_threshold,
        min_weight=args.min_weight
    )

    # Visualize
    if not args.no_visualize:
        graph_path = output_dir / "universal_causal_graph.png"
        aggregator.draw_universal_graph(
            save_path=str(graph_path),
            show_scene_count=True
        )

    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Universal graph: {len(universal_graph.nodes())} nodes, {len(universal_graph.edges())} edges")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
