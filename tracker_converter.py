import pandas as pd
from tigramite import data_processing as tigr
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from networkx import DiGraph, draw
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import numpy as np
import scipy.stats as stats

def load_parquet(path) -> pd.DataFrame:
    df = pd.read_parquet(path=path, engine="pyarrow")
    return df


def add_edges(global_edges,edge_stats):
    for (src,tgt),data in edge_stats.items():
        if data["count"] < 10:  #must appear in >= 10 scenes
            continue

        chi = -2 * np.sum(np.log(data["pvals"]))
        p_global = 1 - stats.chi2.cdf(chi,2 * len(data["pvals"]))

        avg_weight = np.mean(data["weights"])

        if p_global < 0.01:
            global_edges.append((src,tgt,p_global,avg_weight))


    return global_edges

def draw_graph(G,var_names):

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

    for u,v,data in G.edges(data=True):    
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
        G, pos,
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

# Read from all scences( saved trcaker results from all scences) 
# Learn a joint graph from all scences using Tigramite.
# RQ3: use the learnt graph to predict which scences will lead to an accident and evaluate this against the Benchmark methodologies.
if __name__ == "__main__":

    edge_stats = defaultdict(lambda:{
        "count":0,
        "pvals": [],
        "weights": [] 
        })

    var_names = ["x","y","w","h","vx","vy"]
    dir_path = "C:/Users/project_location/tracked_images/results/*.parquet"

    for parquet in glob.glob(dir_path):
        df = load_parquet(parquet)

        for tid in df.track_id.unique():
            obj = df[df.track_id == tid].sort_values("frame")

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


    global_edges = add_edges(global_edges=[],
                             edge_stats=edge_stats)
   
   
   #Build Global Graph
    G = DiGraph()

    for src,tgt,p,w in global_edges:
        G.add_edge(f"{src}(t-1)",f"{tgt}(t)"
                   ,p=p
                   ,weight=w)


    draw_graph(G=G,var_names=var_names)

         