from collections import defaultdict
import numpy as np
import pandas as pd
from tigramite import data_processing as tigr
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from networkx import DiGraph, draw
import scipy.stats as stats
import matplotlib.pyplot as plt
import re


# Domain knowledge constraints for causal graph filtering
FORBIDDEN_EDGES = {
    ("x", "ttc_proxy"),
    ("y", "ttc_proxy"),
    ("min_distance_t", "proximity"),
    ("mean_rel_speed_t", "risk_speed"),
    ("min_ttc_t", "ttc_proxy"),
}

# Expected causal directions for validation
EXPECTED_DIRECTIONS = {
    ("vx", "x"): "positive",
    ("vy", "y"): "positive",
    ("proximity", "ttc_proxy"): "negative",
    ("risk_speed", "ttc_proxy"): "negative",
}

NODE_LAYERS = {
    # Observables
    "obj_count": 0, "ped_count": 0,
    "min_distance_proxy": 0, "max_closing_rate": 0,
    "q10_risk_speed": 0, "mean_ttc_rate": 0,
    "min_ttc_eff": 0, "ego_speed": 0, "ego_accel": 0,

    # Intermediate factors
    "collision_imminence": 1,
    "behavioural_risk": 1,
    "environmental_hazard": 1,

    # (Optional) risk output if you add it later
    "accident_risk": 2,
}


class FeatureExtractor:
    def __init__(self, track_path: str, env_path: str, tau_max: int = 1,
                 pc_alpha: float = 0.005, fdr_q: float = 0.01,
                 min_effect: float = 0.25):
        self.track_df = pd.read_parquet(track_path, engine="pyarrow").sort_values(["frame", "track_id"])
        self.env_df = pd.read_parquet(env_path, engine="pyarrow") if env_path else None
        self.tau_max = int(tau_max)
        self.pc_alpha = float(pc_alpha)
        self.fdr_q = float(fdr_q)
        self.min_effect = float(min_effect)
        # self.min_support = float(min_support)
        self.graph = DiGraph()
        # print(f" Initializing: /n {self.track_df} /n AND /n {self.env_df}")

    def extract_edges(self) -> pd.DataFrame:
        
        frame_table = self.build_frame_table()
        # return self.track_df
    
        var_names = [
            # observables
            "obj_count","ped_count","min_distance_proxy",
            "q10_risk_speed","ego_speed","ego_accel",
            "q05_ttc_eff", "q90_closing_rate",
            # intermediate nodes 
            #"collision_imminence","behavioural_risk","environmental_hazard"
            # Removed nodes "mean_ttc_rate","min_ttc_eff" "max_closing_rate",
        ]

        X = frame_table[var_names].to_numpy(dtype=float)
        tig_df = tigr.DataFrame(X, var_names=var_names)
        pcmci = PCMCI(dataframe=tig_df, cond_ind_test=ParCorr())

        results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=self.pc_alpha)

        graph = results["graph"]
        pmat = results["p_matrix"]
        vmat = results["val_matrix"]

        rows = []
        for i, src in enumerate(var_names):
            for j, tgt in enumerate(var_names):
                for tau in range(1, self.tau_max + 1):
                    if graph[i, j, tau] != "-->":
                        continue
                    if not self._allowed_edge(src, tgt):
                        continue
                    if (src, tgt) in FORBIDDEN_EDGES:
                        continue

                    p = float(pmat[i, j, tau])
                    w = float(abs(vmat[i, j, tau]))
                    rows.append({"src": src, "tgt": tgt, "tau": tau, "p": p, "weight": w})

        edge_df = pd.DataFrame(rows)
        if edge_df.empty:
            return edge_df

        # FDR control (multiple testing)
        edge_df["p_fdr"] = self._bh_fdr(edge_df["p"].to_numpy())

        # effect size filtering
        edge_df = edge_df[edge_df["weight"] >= self.min_effect]

        # significance after FDR
        edge_df = edge_df[edge_df["p_fdr"] <= self.fdr_q].reset_index(drop=True)

        return edge_df

            
    def add_edges(self, edge_df: pd.DataFrame, min_count: int = 0):
        if edge_df is None or edge_df.empty:
            return []

        var_names = sorted(set(edge_df["src"]).union(set(edge_df["tgt"])))
        edge_count= 0
        for _, r in edge_df.iterrows():
            src, tgt, tau = r["src"], r["tgt"], int(r["tau"])
            p = float(r.get("p_fdr", r["p"]))
            w = float(r["weight"])
            edge_count +=1
            self.graph.add_edge(f"{src}(t-{tau})", f"{tgt}(t)", p=p, weight=w)
        print(f"GRAPH with  {len(var_names)} VARNAMES")

        print(f"GRAPH with {edge_count} edges")


        return var_names



    def draw_graph(self, var_names=None, save_path=None):
        G = self.graph
        if G is None or G.number_of_edges() == 0:
            print("No edges to draw.")
            return

        # Parse nodes like "name(t)" or "name(t-2)"
        pat = re.compile(r"^(?P<name>.+)\(t(?:-(?P<lag>\d+))?\)$")

        def parse_node(n: str):
            m = pat.match(str(n))
            if not m:
                return None, None
            name = m.group("name")
            lag_s = m.group("lag")
            lag = int(lag_s) if lag_s is not None else 0
            return name, lag

        # Determine lags present in graph nodes
        lag_to_nodes = {}
        all_base = set()

        for n in G.nodes():
            base, lag = parse_node(n)
            if base is None:
                continue
            all_base.add(base)
            lag_to_nodes.setdefault(lag, []).append(n)

        if not lag_to_nodes:
            print("Could not parse any nodes of form 'name(t-#)'.")
            return

        # If var_names provided, restrict to those bases
        if var_names is not None:
            var_set = set(var_names)
            lag_to_nodes = {
                lag: [n for n in nodes if (parse_node(n)[0] in var_set)]
                for lag, nodes in lag_to_nodes.items()
            }
            lag_to_nodes = {lag: nodes for lag, nodes in lag_to_nodes.items() if nodes}

        # Sort lags descending so columns go: t-k ... t-1 ... t
        lags_sorted = sorted(lag_to_nodes.keys(), reverse=True)

        # For consistent row ordering, order by base name
        base_sorted = sorted(all_base if var_names is None else set(var_names))

        # Build positions: one column per lag
        pos = {}
        x_gap = 4.0  # horizontal gap between lag columns

        for col, lag in enumerate(lags_sorted):
            x = col * x_gap
            # place nodes in same vertical order across columns based on base name
            nodes = lag_to_nodes[lag]
            # map base -> node string for this lag
            base_to_node = {}
            for n in nodes:
                b, _ = parse_node(n)
                base_to_node[b] = n

            for row, base in enumerate(base_sorted):
                if base in base_to_node:
                    pos[base_to_node[base]] = (x, -row)

        # Ensure every node has a position (fallback)
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            # stack missing nodes below
            start_y = -(len(base_sorted) + 2)
            for i, n in enumerate(missing):
                pos[n] = (0.0, start_y - i)

        # Edge styling
        edge_colours = []
        edge_widths = []

        for u, v, data in G.edges(data=True):
            p = data.get("p", 1.0)
            w = data.get("weight", 1.0)

            try:
                p = float(p)
            except Exception:
                p = 1.0
            try:
                w = float(w)
            except Exception:
                w = 1.0

            if p < 0.000001:
                edge_colours.append("red")
            elif p < 0.005:
                # edge_colours.append("orange")
                continue
            else:
                continue

            edge_widths.append(max(0.5, 4.0 * abs(w)))

        plt.figure(figsize=(14, 8))
        draw(
            G, pos,
            with_labels=True,
            node_color="lightblue",
            node_size=3000,
            width=edge_widths,
            edge_color=edge_colours,
            font_size=10,
            arrows=True,
            connectionstyle="arc3,rad=0.08", 
        )

        plt.title("Learned Causal Graph (multi-lag)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"Saved graph to: {save_path}")
        else:
            plt.show()


    def _build_observable_timeseries(self) -> pd.DataFrame:
        df = self.track_df_tracks.copy()

        # Ensure expected columns exist; adjust if your schema differs
        # bbox stored as list/tuple [x1,y1,x2,y2]
        bbox = np.vstack(df["bbox"].apply(lambda b: np.array(b, dtype=float)).to_numpy())
        df["x1"], df["y1"], df["x2"], df["y2"] = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
        df["cx"] = (df["x1"] + df["x2"]) / 2.0
        df["cy"] = (df["y1"] + df["y2"]) / 2.0
        df["w"]  = (df["x2"] - df["x1"]).clip(lower=1.0)
        df["h"]  = (df["y2"] - df["y1"]).clip(lower=1.0)

        # Sort for derivatives
        df = df.sort_values(["track_id", "frame"])

        # Ego-motion columns: you pass (ego_dx, ego_dy, ego_speed, ego_accel) into tracker.update
        # So tracks parquet should contain these per detection/track state.
        # If not, you must add them in tracker.update output.
        for col in ["ego_dx", "ego_dy", "ego_speed", "ego_accel"]:
            if col not in df.columns:
                df[col] = 0.0

        # Per-track velocities in image coords (compensated by ego-motion)
        df["vx"] = df.groupby("track_id")["cx"].diff().fillna(0.0) - df["ego_dx"]
        df["vy"] = df.groupby("track_id")["cy"].diff().fillna(0.0) - df["ego_dy"]

        # A simple "distance-to-bottom-center" proxy (ego-centric driving)
        # Use per-frame image size if available; otherwise approximate by median bbox coords scale.
        # Better: store frame_w, frame_h once in metadata/env.
        frame_w = float(self.track_df_env.get("frame_w", pd.Series([1920])).iloc[0]) if len(self.track_df_env) else 1920.0
        frame_h = float(self.track_df_env.get("frame_h", pd.Series([1080])).iloc[0]) if len(self.track_df_env) else 1080.0

        df["rel_distance"] = np.sqrt((df["cx"] - frame_w/2.0)**2 + (df["cy"] - frame_h)**2)

        # Closing speed and TTC proxy (clip/eps to avoid blowups)
        df["closing_speed"] = -df.groupby("track_id")["rel_distance"].diff().fillna(0.0)
        eps = 1e-3
        df["ttc_proxy"] = (df["rel_distance"] / (df["closing_speed"].clip(lower=eps))).clip(upper=1e4)

        # Lane offset proxy: horizontal displacement
        df["lane_offset"] = df["cx"] - frame_w/2.0
        df["lateral_speed"] = df["vx"]

        # Aggregate to frame-level observables (scene state), not track-level noise
        frame_obs = df.groupby("frame").agg(
            obj_count=("track_id", "nunique"),
            mean_rel_distance=("rel_distance", "mean"),
            min_rel_distance=("rel_distance", "min"),
            mean_closing_speed=("closing_speed", "mean"),
            max_closing_speed=("closing_speed", "max"),
            min_ttc=("ttc_proxy", "min"),
            mean_lane_offset=("lane_offset", "mean"),
            std_lane_offset=("lane_offset", "std"),
            mean_ego_speed=("ego_speed", "mean"),
            mean_ego_accel=("ego_accel", "mean"),
        ).fillna(0.0)

        # Merge environment features (your EnvironmentBuilder output)
        # Assuming env_df has "frame" column; if not, add it.
        if "frame" in self.track_df_env.columns:
            frame_obs = frame_obs.merge(self.track_df_env, on="frame", how="left").fillna(0.0)

        return frame_obs.reset_index(drop=False)
    

    def build_frame_table(self):
        # df = self.track_df.copy()

        vehicles = self.track_df[self.track_df["class_id"].isin([0,1,2,4])]
        pedestrians = self.track_df[self.track_df["class_id"].isin([3])]

        # print (f"Vehicles: {vehicles}")
        # print (f"Pedestrians : {pedestrians}")

        TTC_MAX = 999.0

        for col in ["ttc_proxy", "ttc_relative"]:
            if col in vehicles.columns:
                vehicles.loc[vehicles[col]>= TTC_MAX,col]= np.nan
         
        vehicles["ttc_confidence"]= vehicles["ttc_confidence"].clip(lower=1e-6, upper=1e6)

        conf_gate = 0.25
        vehicles["ttc_ok"] = vehicles["ttc_confidence"] >= conf_gate

        vehicles["ttc_eff"] = vehicles["ttc_proxy"] / vehicles["ttc_confidence"]
        vehicles.loc[~vehicles["ttc_ok"], "ttc_eff"] = np.nan

        # def confidence_min_ttc(g):
        #     # weight TTC by confidence: prefer confident TTCs
        #     # Use effective TTC = ttc / conf to penalize low confidence
        #     eff = g["ttc_proxy"] / g["ttc_confidence"].clip(lower=1e-3)
        #     #print(f"eff : {eff}")
        #     return float(np.min(eff)) if len(eff) else 999.0

        def get_quantile(x, p):
            x = x.dropna()
            return float(x.quantile(p)) if len(x) else np.nan 

        vehicle_agg = vehicles.groupby("frame", sort=True)

        # Frame-level observables
        frame_obs = vehicle_agg.agg(
            obj_count=("track_id", "nunique"),
            min_distance_proxy=("distance_proxy", "min"),
            # max_closing_rate=("closing_rate", "max"),
            q90_closing_rate=("closing_rate", lambda s: get_quantile(s, 0.90)),
            # risk context
            q10_risk_speed=("risk_speed", lambda x: get_quantile(x,0.1)),
            #TTC_proxies
            #  mean_ttc_rate=("ttc_rate", "mean"), #can't use mean. I have ttc=999
            q10_ttc_proxy=("ttc_proxy", lambda s: get_quantile(s,0.10)),
            q05_ttc_eff=("ttc_eff",   lambda s: get_quantile(s, 0.05)),

            #cennters
            mean_speed=("speed", "median"), # formerly mean 
            mean_rel_speed=("ttc_relative", "mean"), # formerly mean 

            #ego speed
            ego_speed=("ego_speed", "mean"),
            ego_accel=("ego_accel", "mean"),
        ).reset_index()

        ttc_valid = vehicle_agg.apply(lambda x: np.mean(np.isfinite(x["ttc_proxy"].to_numpy()))).rename("ttc_valid_frac")
        ttc_conf_ok = vehicle_agg.apply(lambda x: np.mean(x["ttc_ok"].to_numpy())).rename("ttc_conf_ok_frac")
        frame_obs = frame_obs.merge(ttc_valid.reset_index(), on="frame", how="left")
        frame_obs = frame_obs.merge(ttc_conf_ok.reset_index(), on="frame", how="left")


        # # Add confidence-weighted min TTC (more robust)
        # ttc_conf = vehicles.groupby("frame").apply(confidence_min_ttc).rename("min_ttc_eff").reset_index()
        # frame_obs = frame_obs.merge(ttc_conf, on="frame", how="left").fillna(999.0)

        # print(f"frame_obs desc: \n {frame_obs[['q10_risk_speed', 'q10_ttc_proxy']].describe()}")
        # print (f"frame_obs: \n {frame_obs}")
         
        if len(pedestrians):
            ped_obs = pedestrians.groupby("frame", sort=True).agg(
                ped_count=("track_id", "nunique"),
                min_ped_distance=("distance_proxy", "min"),
            ).reset_index()
            frame_obs = frame_obs.merge(ped_obs, on="frame", how="left")
        else:
            frame_obs["ped_count"] = 0
            frame_obs["min_ped_distance"] = np.nan

        #frame_obs = frame_obs.merge(ped_obs, on="frame", how="left").fillna({"ped_count":0, "min_ped_distance":999.0})

        # Merge environment features if available 
        if self.env_df is not None and "frame" in self.env_df.columns:
            frame_obs = frame_obs.merge(self.env_df, on="frame", how="left")

        # Intermediate proxy nodes 
        # frame_obs["collision_imminence"] = (
        #     (1.0 / frame_obs["min_ttc_eff"].clip(lower=1e-3)) +
        #     frame_obs["max_closing_rate"].clip(lower=0.0) +
        #     frame_obs["q10_risk_speed"].clip(lower=0.0)
        # )

        # frame_obs["behavioural_risk"] = (
        #     frame_obs["ego_accel"].abs() +
        #     frame_obs["mean_ttc_rate"].abs()  
        # )

        # if "blur" in frame_obs.columns:
        #     frame_obs["environmental_hazard"] = frame_obs["blur"]
        # elif "illumination" in frame_obs.columns:
        #     frame_obs["environmental_hazard"] = (1.0 / (frame_obs["illumination"].clip(lower=1e-3)))
        # else:
        #     frame_obs["environmental_hazard"] = frame_obs["ego_accel"].abs()
        frame_obs["min_ped_distance"] = frame_obs["min_ped_distance"].fillna(TTC_MAX)
        frame_obs["q10_ttc_proxy"] = frame_obs["q10_ttc_proxy"].fillna(TTC_MAX)
        frame_obs["q05_ttc_eff"] = frame_obs["q05_ttc_eff"].fillna(TTC_MAX)

        smooth_cols = ["q90_closing_rate", "q10_risk_speed", "q10_ttc_proxy", "q05_ttc_eff",
                   "obj_count", "ped_count", "ego_speed", "ego_accel"]


        for c in smooth_cols:
            if c in frame_obs.columns:
                frame_obs[c] = frame_obs[c].rolling(window=3, center=True, min_periods=1).median()

        frame_obs["collision_imminence"] = (
        (1.0 / np.clip(frame_obs["q05_ttc_eff"], 1e-3, None)) +
        np.clip(frame_obs["q90_closing_rate"], 0.0, None) +
        np.clip(frame_obs["q10_risk_speed"], 0.0, None)
         )

        frame_obs["behavioural_risk"] = (
            frame_obs["ego_accel"].abs()
            # If you reintroduce TTC-rate, use a robust version, not mean of noisy differences.
        )

        if "blur" in frame_obs.columns:
            frame_obs["environmental_hazard"] = frame_obs["blur"]
        elif "illumination" in frame_obs.columns:
            frame_obs["environmental_hazard"] = 1.0 / frame_obs["illumination"].clip(lower=1e-3)
        else:
            frame_obs["environmental_hazard"] = frame_obs["ego_accel"].abs()
                                                        
        return frame_obs.sort_values("frame").reset_index(drop=True)

    def _allowed_edge(self, src, tgt):
        # allow within-level temporal edges and lower->higher
        return NODE_LAYERS.get(src, 0) <= NODE_LAYERS.get(tgt, 0)
    
    def _bh_fdr(self, pvals: np.ndarray) -> np.ndarray:
        p = np.asarray(pvals, dtype=float)
        n = p.size
        order = np.argsort(p)
        ranked = p[order]
        thresh = (np.arange(1, n+1) / n) * self.fdr_q
        passed = ranked <= thresh

        out = np.ones_like(p)
        if np.any(passed):
            k = np.max(np.where(passed)[0])
            cutoff = ranked[k]
            out[p <= cutoff] = p[p <= cutoff]
        return out

    @staticmethod
    def get_causal_graph_for_dbn(graph: DiGraph) -> DiGraph:
        """
        Convert FeatureExtractor.graph (nodes like 'ttc_proxy_d(t-1)' -> 'risk_speed_d(t)')
        into a pgmpy-friendly 2-slice DBN edge graph:
            nodes: ('var_d', 0) and ('var_d', 1)
            edges: (('src_d', 0), ('tgt_d', 1))  for lag>=1
        Keeps edge attributes: p, weight, lag.
        Collapses duplicates by keeping the edge with best (lowest p, then highest weight).
        """
        # Accept "x(t)" or "x(t-2)" or "x(t-1)"
        pat = re.compile(r"^(?P<name>.+)\(t(?:-(?P<lag>\d+))?\)$")

        def parse(n: str):
            m = pat.match(str(n))
            if not m:
                return None, None
            name = m.group("name")
            lag_s = m.group("lag")
            lag = int(lag_s) if lag_s is not None else 0
            return name, lag

        def p_val(x):
            try:
                return float(x)
            except Exception:
                return float("inf")

        def w_val(x):
            try:
                return float(x)
            except Exception:
                return float("-inf")

        # Best edge per (src_base, tgt_base) regardless of lag
        best = {}  # key -> dict(attrs)

        for u, v, data in graph.edges(data=True):
            src, src_lag = parse(u)
            tgt, tgt_lag = parse(v)

            # Only keep edges that look like (t-k) -> (t)
            if src is None or tgt is None:
                continue
            if not (src_lag >= 1 and tgt_lag == 0):
                continue

            lag = src_lag
            p = data.get("p", None)
            w = data.get("weight", None)

            key = (src, tgt)
            cand = {"lag": lag, "p": p, "weight": w}

            if key not in best:
                best[key] = cand
            else:
                prev = best[key]
                # Prefer lower p, then higher weight; if p missing, weight decides
                prev_score = (p_val(prev["p"]), -w_val(prev["weight"]))
                cand_score = (p_val(cand["p"]), -w_val(cand["weight"]))
                if cand_score < prev_score:
                    best[key] = cand

        # Build 2-slice pgmpy-style graph
        dbn_edges = DiGraph()

        for (src, tgt), info in best.items():
            src_node = (src, 0)
            tgt_node = (tgt, 1)

            dbn_edges.add_node(src_node)
            dbn_edges.add_node(tgt_node)

            dbn_edges.add_edge(
                src_node,
                tgt_node,
                lag=int(info["lag"]),
                p=info["p"],
                weight=info["weight"],
            )

        return dbn_edges
