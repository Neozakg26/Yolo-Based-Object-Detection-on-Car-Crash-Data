import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from training.config_loader import ConfigLoader
from execute.global_graph import draw_global_graph_time_lag

from explainability.feature_extractor import FeatureExtractor

MISSING = -9999.0

# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")

BASE_PATH = config.paths["local_base"]

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    adj = np.empty(n, dtype=float)
    adj[order] = pvals[order] * n / ranks
    # enforce monotonicity
    adj_rev = np.minimum.accumulate(adj[order][::-1])[::-1]
    adj[order] = np.clip(adj_rev, 0.0, 1.0)
    return adj

def infer_scene_id(path: str) -> str:
    base = os.path.basename(path)
    # expects like 000001_tracks.parquet
    return base.split("_")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True,
                    help="Directory containing *_tracks.parquet and *_env.parquet files")
    ap.add_argument("--out_path", type=str, required=True,
                    help="Where to save the global graph pickle (.pkl)")

    # PCMCI params 
    ap.add_argument("--tau_max", type=int, default=1)
    ap.add_argument("--pc_alpha", type=float, default=0.01)
    ap.add_argument("--fdr_q", type=float, default=0.01)
    ap.add_argument("--min_effect", type=float, default=0.20)

    # variables to use (keep to observables; avoid deterministic “intermediate” nodes)
    ap.add_argument("--var_names", type=str, default="obj_count,ped_count,min_distance_proxy,q90_closing_rate,q10_risk_speed,q10_ttc_proxy,q05_ttc_eff,ego_speed,ego_accel,ttc_valid_frac,ttc_conf_ok_frac")

    args = ap.parse_args()
    var_names = [v.strip() for v in args.var_names.split(",") if v.strip()]

    track_files = sorted(glob.glob(os.path.join(args.results_dir, "*_tracks.parquet")))
    if not track_files:
        raise FileNotFoundError(f"No *_tracks.parquet found in {args.results_dir}")

    Xs, masks = [], []
    used_scenes = []

    for track_path in track_files:
        scene_id = infer_scene_id(track_path)
        env_path = os.path.join(args.results_dir, f"{scene_id}_env.parquet")
        if not os.path.exists(env_path):
            env_path = None # allow missing env;

        fe = FeatureExtractor(
            track_path=track_path,
            env_path=env_path,
            tau_max=args.tau_max,
            pc_alpha=args.pc_alpha,
            fdr_q=args.fdr_q,
            min_effect=args.min_effect,
        )

        ft = fe.build_frame_table()

        print (f"frame table  \n {ft}")
        # ensure required columns exist
        for c in var_names:
            if c not in ft.columns:
                ft[c] = np.nan

        X = ft[var_names].to_numpy(dtype=float)
        print (f"x  \n {X}")


        bad = ~np.isfinite(X)
        X = np.where(bad, MISSING, X)

        Xs.append(X)
        masks.append(bad)
        used_scenes.append(scene_id)

    # data = np.stack(Xs, axis=0)    # (M, T, N)
    # mask = np.stack(masks, axis=0)
    lengths = [X.shape[0] for X in Xs]
    Tmax = max(lengths)
    N = Xs[0].shape[1]
    M = len(Xs)

    data = np.full((M, Tmax, N), MISSING, dtype=float)
    mask = np.ones((M, Tmax, N), dtype=bool)  # True = missing

    for m, X in enumerate(Xs):
        Ti = X.shape[0]
        data[m, :Ti, :] = X
        mask[m, :Ti, :] = False  # observed where we wrote real values

    df_multi = pp.DataFrame(
        data=data,
        mask=mask,
        missing_flag=MISSING,
        var_names=var_names,
        analysis_mode="multiple"
    )

    pcmci = PCMCI(dataframe=df_multi, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=args.pc_alpha)

    graph = results["graph"]
    pmat = results["p_matrix"]
    vmat = results["val_matrix"]

    rows = []
    N = len(var_names)
    for i, src in enumerate(var_names):
        for j, tgt in enumerate(var_names):
            for tau in range(1, args.tau_max + 1):
                if graph[i, j, tau] != "-->":
                    continue
                p = float(pmat[i, j, tau])
                w = float(abs(vmat[i, j, tau]))
                rows.append({"src": src, "tgt": tgt, "tau": tau, "p": p, "weight": w})

    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        G = nx.DiGraph()
        payload = {"graph": G, "edge_df": edge_df, "var_names": var_names, "scenes": used_scenes}
        with open(args.out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"No edges discovered. Saved empty graph to {args.out_path}")
        return

    edge_df["p_fdr"] = bh_fdr(edge_df["p"].to_numpy())
    edge_df = edge_df[edge_df["weight"] >= args.min_effect]
    edge_df = edge_df[edge_df["p_fdr"] <= args.fdr_q].reset_index(drop=True)

    # Build a NetworkX DiGraph with lag attributes
    G = nx.DiGraph()
    for _, r in edge_df.iterrows():
        G.add_edge(r["src"], r["tgt"], tau=int(r["tau"]), p=float(r["p"]), p_fdr=float(r["p_fdr"]), weight=float(r["weight"]))

    payload = {"graph": G, "edge_df": edge_df, "var_names": var_names, "scenes": used_scenes}
    with open(args.out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved global PCMCI graph: {args.out_path}")
    print(f"Scenes pooled: {len(used_scenes)}")
    print(f"Edges kept after FDR+effect: {len(edge_df)}")

    draw_global_graph_time_lag(G=G,var_names=var_names,save_path=f"{BASE_PATH}/global_causal_.png")

if __name__ == "__main__":
    main()
