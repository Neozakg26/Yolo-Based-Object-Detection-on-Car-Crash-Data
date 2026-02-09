from ultralytics import YOLO
from tracking.deepsort_tracker import DeepSortTracker
from tracking.track_runner import TrackingRunner
from training.config_loader import ConfigLoader
from explainability.metadata import MetaData
from explainability.environment_builder import EnvironmentBuilder
import pandas as pd
import argparse
import re
# ---------- Phase 1 ----------
# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
# parser.add_argument("--skip_viz", action="store_true", help="Skip graph visualization")
# parser.add_argument("--skip_dbn", action="store_true", help="Skip hierarchical DBN risk assessment")
# parser.add_argument("--tau_max", type=int, default=2, help="Max time lag for PCMCI")
# parser.add_argument("--inference_method", type=str, default="supervised",
#                     choices=["supervised", "belief_propagation", "variable_elimination"],
#                     help="Inference method: supervised (recommended), belief_propagation, or variable_elimination")
args = parser.parse_args()


# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")

BASE_PATH = config.paths["cluster_base"]
META_PATH = f"{BASE_PATH}/Crash_Table.csv"

# ---------- INIT MODELS ----------
detector = YOLO(config.paths["cluster_model"])
tracker = DeepSortTracker()
runner = TrackingRunner(detector, tracker)

# ---------- LOAD METADATA ----------
scene_id = re.search(r'\d+', args.path).group()
meta = MetaData(META_PATH, scene_id)


# ---------- TRACK ----------
all_tracks = runner.run(f"{BASE_PATH}/{args.path}", metadata=meta.metadata)
#print(f"all tracks \n {all_tracks}")
df_tracks = pd.DataFrame(all_tracks).sort_values(["track_id", "frame"])

# print(f"DF tracks \n {df_tracks}") 
# print("DF TRACKS") #DEBUG 
# print(f"{df_tracks.head}") #DEBUG 
tracks_path = f"{BASE_PATH}/results/{scene_id}_tracks.parquet"
df_tracks.to_parquet(tracks_path, index=False,engine="pyarrow")

print(f"Tracks saved {tracks_path}")

# ---------- BUILD ENVIRONMENT ----------
env_df = EnvironmentBuilder.build(df_tracks)
env_path = f"{BASE_PATH}/results/{scene_id}_env.parquet"
env_df.to_parquet(env_path, index=False, engine="pyarrow")

print(f"Environment features saved: {env_path}")

