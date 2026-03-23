from explainability.feature_extractor import FeatureExtractor
from explainability.hierarchical_dbn import AccidentRiskAssessor
from explainability.metadata import MetaData
from training.config_loader import ConfigLoader
import pandas as pd
from pathlib import Path
import argparse
import re
import pickle
from networkx import DiGraph
# ---------- PHASE2  ----------

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--track_path", type=str, required=True)
parser.add_argument("--env_path", type=str, required=True)

parser.add_argument("--skip_viz", action="store_true", help="Skip graph visualization")
parser.add_argument("--tau_max", type=int, default=2, help="Max time lag for PCMCI")
parser.add_argument("--inference_method", type=str, default="supervised",
                    choices=["supervised", "belief_propagation", "variable_elimination"],
                    help="Inference method: supervised (recommended), belief_propagation, or variable_elimination")
args = parser.parse_args()

# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")

BASE_PATH = config.paths["cluster_base"]

META_PATH = f"{BASE_PATH}/Crash_Table.csv"

df_tracks = Path(args.track_path)
env_df = Path(args.env_path)


# ---------- LOAD METADATA ----------
scene_id = re.search(r'\d+', args.track_path ).group()
meta = MetaData(META_PATH, scene_id)

# ---------- FEATURE EXTRACTION FOR CAUSAL ----------
features = FeatureExtractor(track_path=args.track_path, env_path=args.env_path, tau_max=args.tau_max)

edge_stats =features.extract_edges()
#print(f"EDGE STATS: {edge_stats.items()}")
var_names = features.add_edges(edge_stats, min_count=0)
#print(f"EdGE STATS :{edge_stats}")

# print(f"EdGE STATS DF :{edge_stats_df.head}")
# Save edge statistics for later aggregation across scenes
edge_stats_path = f"{BASE_PATH}/results/{scene_id}_edge_stats.parquet"

edge_stats.to_parquet(edge_stats_path, index=False, engine="pyarrow")

print(f"Edge statistics saved: {edge_stats_path}")

# Get PCMCI causal graph for DBN structure # graph file is _causal_graph
pcmci_graph = FeatureExtractor.get_causal_graph_for_dbn(graph=features.graph)

# Save the causal graph
graph_path = f"{BASE_PATH}/results/{scene_id}_causal_graph.pkl"
with open(f"{graph_path}","wb") as f:
    pickle.dump(pcmci_graph,f)

print(f"Graph .pkl file  saved: {graph_path}")
# Print summary
#features.print_graph_summary()

# Visualize (optional)
if not args.skip_viz:
    fig_path = f"{BASE_PATH}/results/{scene_id}_causal_graph.png"
    features.draw_graph(var_names=var_names, save_path=fig_path)
