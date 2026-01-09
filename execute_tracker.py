from ultralytics import YOLO
from tracking.deepsort_tracker import DeepSortTracker
from tracking.track_runner import TrackingRunner
from tracking.feature_extractor import FeatureExtractor
from training.config_loader import ConfigLoader
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

config = ConfigLoader.load("config.yaml")

#Load Detector 

detector = YOLO(config.model["best"])  
print(f"Loaded Model: {config.model['best']}")
#Initialise Tracker 
tracker = DeepSortTracker()
runner = TrackingRunner(detector, tracker)

# Run Tracker on a specific Scene
base_path = "\\datasets\\nmaja\\bdd100k\\images\\seg_track_20\\train"

all_tracks = runner.run(f"{base_path}\\{args.path}")
print(f"Tracking Completed: {args.path}")
#save Tracker results as Parquet File 
df = pd.DataFrame(all_tracks)
df.to_parquet(f"{base_path}\\results\\{args.path}.parquet", engine="pyarrow", index=False)


# #Extract tracked Features
# features = FeatureExtractor(
#     all_tracks=all_tracks
# )
# timeseries = []
# frame_shape = (720, 1280, 3)
# for frame_idx, track in enumerate(all_tracks):
#     row = features.build_timeseries_row(track=track,
#                                         frame_idx=frame_idx,
#                                         frame_shape=frame_shape)
#     timeseries.append(row)

# df = pd.DataFrame(timeseries)
# dynotears_input = df.values  # DYNOTEARS input