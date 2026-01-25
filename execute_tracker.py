from ultralytics import YOLO
from tracking.deepsort_tracker import DeepSortTracker
from tracking.track_runner import TrackingRunner
from explainability.feature_extractor import FeatureExtractor
from training.config_loader import ConfigLoader
from explainability.metadata import MetaData
import pandas as pd
import argparse
import re

def save_results(all_tracks, path):
    df = convert_from_df(all_tracks=all_tracks)
    df.to_parquet(path, engine="pyarrow", index=False)

def convert_from_df(all_tracks):
    df = pd.DataFrame(all_tracks)
    # sort by track id and frame
    df = df.sort_values(["track_id","frame"])
    
    return df

#Load Args
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
# parser. add_argument("--scene_id", type=str, required=True)

args = parser.parse_args()

#initialise Global Variables
config = ConfigLoader.load("config.yaml")
BASE_PATH = "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images"
detector = YOLO(config.model["best"])  
tracker = DeepSortTracker() 
runner = TrackingRunner(detector, tracker)

#START TRACKING 
#LOAD ACCIDENT METADATA
metdata_path  = "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images/Crash_Table.csv"
scene_id = re.search(r'\d+',args.path).group()
mt_df = MetaData(metdata_path,scene_id= scene_id)
limit = int(mt_df.metadata.get('accident_start_frame'))


all_tracks = runner.run(f"{BASE_PATH}/{args.path}",limit=limit)
pq_path= f"C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/tracked_images/results/{args.path}.parquet"


save_results(all_tracks= all_tracks,
             path=pq_path)
print(f"Tracking Completed: {args.path} and Saved: {pq_path}")

# Tracking Ended. Output -> C0001.parquet file 


#Extract Features
features = FeatureExtractor(
    path=pq_path
)

edge_stats = features.extract_edges()


features.add_edges(edge_stats=edge_stats)

var_names = ["x","y","w","h","vx","vy"]
features.draw_graph(var_names=var_names)
# df = pd.DataFrame(timeseries)
# dynotears_input = df.values  # DYNOTEARS input