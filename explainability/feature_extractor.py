from collections import defaultdict
import numpy as np
import pandas as pd
from tigramite import data_processing as tigr
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from networkx import DiGraph, draw
import scipy.stats as stats
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, path):
        self.df = self.__load_parquet__(path)
        self.track_history = defaultdict(list)  # track_id → list of (frame_idx, cx, cy)
        self.graph = DiGraph()
        
    
    def __load_parquet__(self,path) -> pd.DataFrame:
        df = pd.read_parquet(path=path, engine="pyarrow")
        return df


    def extract_edges(self):
        edge_stats = defaultdict(lambda:{
        "count":0,
        "pvals": [],
        "weights": [] 
        })

        var_names = ["x","y","w","h","vx","vy"]

        for tid in self.df.track_id.unique():
            obj = self.df[self.df.track_id == tid].sort_values("frame")

            if len(obj) < 30:
                continue # too short for causal inference 

            X =  obj[var_names].to_numpy()
            tigramite_df = tigr.DataFrame(X)
            pcmci = PCMCI(
                dataframe=tigramite_df,
                cond_ind_test=ParCorr())
            results  = pcmci.run_pcmci()

            graph =results.get("graph")
            pvals =results.get("p_matrix")
            values =results.get("val_matrix")

            #Count edges 
            for i,src in enumerate(var_names):
                for j,tgt in enumerate(var_names):
                    if graph[i,j,1] == "-->":
                        edge = (src,tgt)
                        edge_stats[edge]["count"] +=1
                        edge_stats[edge]["pvals"].append(pvals[i,j,1])
                        edge_stats[edge]["weights"].append(abs(values[i,j,1]))

            return edge_stats
            
    def add_edges(self,edge_stats):

        for (src,tgt),data in edge_stats.items():
            print(f"inside loop add_edges with{(src,tgt)}and {data}")
            # if data["count"] < 10:  #must appear in >= 10 scenes  ## ONLY VALID WHEN MERGING ALL SCENES
            #     continue

            chi = -2 * np.sum(np.log(data["pvals"]))
            p_global = 1 - stats.chi2.cdf(chi,2 * len(data["pvals"]))

            avg_weight = np.mean(data["weights"])
            
            if p_global < 0.01:
                self.graph.add_edge(f"{src}(t-1)",f"{tgt}(t)"
                   ,p=p_global
                   ,weight=avg_weight)

    def draw_graph(self,var_names):

        plt.figure(figsize=(10,8))
        left = [f"{v}(t-1)" for v in var_names]
        right = [f"{v}(t)" for v in var_names]

        pos = {}

        # left column
        for i, node in enumerate(left):
            pos[node] = (0, -i)

        # right column
        for i, node in enumerate(right):
            pos[node] = (4, -i)


        # Add coloured edges
        edge_colours=[]
        edge_widths=[]

        
        for u,v,data in self.graph.edges(data=True):  
            
            p=data["p"]
            w=data["weight"]

            if p< 0.0001:
                edge_colours.append("red")
            elif p<0.005:
                edge_colours.append("orange")
            else:
                edge_colours.append("blue")

            edge_widths.append(4*w)


        draw(
            self.graph, pos,
            with_labels=True,
            node_color="lightblue",
            node_size=3000,
            width=edge_widths,
            edge_color=edge_colours,
            font_size=10,
            arrows=True
        )

        plt.title("Learned Causal Graph from Object Trajectory")
        plt.show()           


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