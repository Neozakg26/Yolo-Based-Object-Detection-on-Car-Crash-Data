from collections import defaultdict
import numpy as np


class FeatureExtractor:
    def __init__(self,all_tracks):
        self.all_tracks = all_tracks
        self.track_history = defaultdict(list)  # track_id → list of (frame_idx, cx, cy)
        
    
    def __extract_features(self, track, frame_idx, frame_shape):
        vehicles = []
        pedestrians = []

        cls = track.get("class_id")
        x1, y1, x2, y2 = track.get("bbox")
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        self.track_history[track.get("track_id")].append((frame_idx, cx, cy))

        if cls in [0, 1, 2]:  # car, bus, truck
            vehicles.append((track.get("track_id"), cx, cy, x1, y1, x2, y2))
        elif cls == 3:  # person
            pedestrians.append((track.get("track_id"), cx, cy))

        return vehicles, pedestrians
    

    def __compute_speed(self,track_id):
        hist = self.track_history[track_id]
        if len(hist) < 2:
            return 0.0
        (_, x1, y1), (_, x2, y2) = hist[-2:]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def build_timeseries_row(self,track, frame_idx, frame_shape):
        vehicles, pedestrians = self.__extract_features(track, frame_idx, frame_shape)
        speeds = [self.__compute_speed(tid) for tid, *_ in vehicles]
        ped_speeds = [self.__compute_speed(tid) for tid, *_ in pedestrians]

        H, W = frame_shape[:2]

        min_vehicle_dist = min(
            [np.sqrt((cx - W/2)**2 + (cy - H)**2) for _, cx, cy, *_ in vehicles],
            default=0.0
        )

        row = {
            "num_vehicles": len(vehicles),
            "num_pedestrians": len(pedestrians),
            "avg_vehicle_speed": np.mean(speeds) if speeds else 0.0,
            "max_vehicle_speed": np.max(speeds) if speeds else 0.0,
            "avg_ped_speed": np.mean(ped_speeds) if ped_speeds else 0.0,
            "min_vehicle_distance": min_vehicle_dist,
            "vehicle_density": len(vehicles) / (H * W)
        }

        return row