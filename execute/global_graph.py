# explainability/global_graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Sequence
from pathlib import Path

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re

@dataclass
class GlobalGraphArtifact:
    """
    Utility wrapper for a globally-pooled PCMCI graph artifact.

    Supports two common pickle formats:
      1) payload dict: {"graph": nx.DiGraph, "edge_df": pd.DataFrame, ...}
      2) raw nx.DiGraph directly
    """
    graph: nx.DiGraph
    payload: Optional[Dict[str, Any]] = None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GlobalGraphArtifact":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, nx.DiGraph):
            return cls(graph=obj, payload=None)

        if isinstance(obj, dict) and "graph" in obj and isinstance(obj["graph"], nx.DiGraph):
            return cls(graph=obj["graph"], payload=obj)

        raise ValueError(
            f"Unsupported global graph pickle format at {path}. "
            f"Expected nx.DiGraph or dict with key 'graph' (nx.DiGraph). Got: {type(obj)}"
        )

    def print_graph_summary(self, top_k: int = 20) -> None:
        """
        Prints a compact summary similar to a typical features.print_graph_summary().
        """
        G = self.graph
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        print("=" * 60)
        print("GLOBAL PCMCI GRAPH SUMMARY")
        print("=" * 60)
        print(f"Nodes: {n_nodes}")
        print(f"Edges: {n_edges}")

        # basic lag distribution if present
        taus = []
        pvals = []
        pfdrs = []
        weights = []
        for _, _, d in G.edges(data=True):
            if "tau" in d:
                taus.append(int(d["tau"]))
            if "p" in d:
                pvals.append(float(d["p"]))
            if "p_fdr" in d:
                pfdrs.append(float(d["p_fdr"]))
            if "weight" in d:
                weights.append(float(d["weight"]))

        if taus:
            uniq, cnt = np.unique(taus, return_counts=True)
            tau_str = ", ".join(f"tau={t}: {c}" for t, c in zip(uniq, cnt))
            print(f"Lag counts: {tau_str}")

        def _describe(arr, name):
            if not arr:
                return
            a = np.asarray(arr, dtype=float)
            print(f"{name}: min={a.min():.3g}, median={np.median(a):.3g}, max={a.max():.3g}")

        _describe(pvals, "p")
        _describe(pfdrs, "p_fdr")
        _describe(weights, "abs(val)")

        # show top edges by weight (or by smallest p_fdr)
        edges = []
        for u, v, d in G.edges(data=True):
            edges.append((u, v, d))

        if edges:
            if weights:
                edges_sorted = sorted(
                    edges,
                    key=lambda e: float(e[2].get("weight", 0.0)),
                    reverse=True
                )
                print(f"\nTop {min(top_k, len(edges_sorted))} edges by |effect|:")
                for u, v, d in edges_sorted[:top_k]:
                    tau = d.get("tau", "?")
                    w = d.get("weight", None)
                    pf = d.get("p_fdr", None)
                    print(f"  {u} -(t-{tau})-> {v} | weight={w:.3g} p_fdr={pf:.3g}" if (w is not None and pf is not None)
                          else f"  {u} -(t-{tau})-> {v} | attrs={d}")
            elif pfdrs:
                edges_sorted = sorted(edges, key=lambda e: float(e[2].get("p_fdr", 1.0)))
                print(f"\nTop {min(top_k, len(edges_sorted))} edges by smallest p_fdr:")
                for u, v, d in edges_sorted[:top_k]:
                    tau = d.get("tau", "?")
                    pf = d.get("p_fdr", None)
                    print(f"  {u} -(t-{tau})-> {v} | p_fdr={pf:.3g}" if pf is not None
                          else f"  {u} -(t-{tau})-> {v} | attrs={d}")

        print("=" * 60)

    def draw_graph(
        self,
        *,
        title: str = "Global PCMCI Graph",
        figsize: Tuple[int, int] = (14, 9),
        layout: str = "spring",                 # "spring" | "kamada_kawai" | "shell"
        seed: int = 42,
        node_size: int = 1700,
        font_size: int = 9,
        edge_width: float = 1.3,
        arrow_size: int = 14,
        margin_pts: int = 16,                   # IMPORTANT: prevents edges drawing over nodes
        curve_rad: float = 0.10,
        show_edge_labels: bool = True,
        edge_label_mode: str = "tau_p",         # "tau" | "tau_p" | "tau_w" | "all"
        label_top_k: Optional[int] = 60,        # limit edge label clutter
        save_path: Optional[Union[str, Path]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Draw the global graph with edge margins so lines/arrows don't write over nodes.

        Edge labels can show tau / p_fdr / weight (if present on edges).
        """
        G = self.graph

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # --- Layout ---
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed)

        # --- Nodes & labels ---
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)

        # --- Edges (key: margins avoid drawing through nodes) ---
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_width,
            arrows=True,
            arrowsize=arrow_size,
            min_source_margin=margin_pts,
            min_target_margin=margin_pts,
            connectionstyle=f"arc3,rad={curve_rad}",
        )

        # --- Edge labels ---
        if show_edge_labels and G.number_of_edges() > 0:
            # build labels
            labels = {}
            edges = list(G.edges(data=True))

            # rank edges if we want to label only top_k
            if label_top_k is not None and len(edges) > label_top_k:
                # prefer weight; else prefer smallest p_fdr; else first edges
                if any("weight" in d for _, _, d in edges):
                    edges = sorted(edges, key=lambda e: float(e[2].get("weight", 0.0)), reverse=True)[:label_top_k]
                elif any("p_fdr" in d for _, _, d in edges):
                    edges = sorted(edges, key=lambda e: float(e[2].get("p_fdr", 1.0)))[:label_top_k]
                else:
                    edges = edges[:label_top_k]

            for u, v, d in edges:
                tau = d.get("tau", None)
                p = d.get("p", None)
                p_fdr = d.get("p_fdr", None)
                w = d.get("weight", None)

                if edge_label_mode == "tau":
                    lab = f"τ={tau}" if tau is not None else ""
                elif edge_label_mode == "tau_p":
                    # prefer p_fdr; fall back to p
                    p_show = p_fdr if p_fdr is not None else p
                    if tau is not None and p_show is not None:
                        lab = f"τ={tau}, p={p_show:.3g}"
                    elif tau is not None:
                        lab = f"τ={tau}"
                    else:
                        lab = ""
                elif edge_label_mode == "tau_w":
                    if tau is not None and w is not None:
                        lab = f"τ={tau}, w={w:.2g}"
                    elif tau is not None:
                        lab = f"τ={tau}"
                    else:
                        lab = ""
                else:  # "all"
                    parts = []
                    if tau is not None:
                        parts.append(f"τ={tau}")
                    if p_fdr is not None:
                        parts.append(f"pFDR={p_fdr:.2g}")
                    elif p is not None:
                        parts.append(f"p={p:.2g}")
                    if w is not None:
                        parts.append(f"w={w:.2g}")
                    lab = ", ".join(parts)

                if lab:
                    labels[(u, v)] = lab

            if labels:
                nx.draw_networkx_edge_labels(
                    G, pos, ax=ax,
                    edge_labels=labels,
                    font_size=max(7, font_size - 2),
                    bbox=dict(alpha=0.0, pad=0.2),
                    rotate=False,
                )

        ax.set_title(title)
        ax.set_axis_off()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        return ax


# ---- Convenience function to mirror features.draw_graph() style usage ----

def draw_global_graph(
    global_graph_pkl: Union[str, Path],
    **kwargs
) -> plt.Axes:
    """
    One-liner convenience:
        draw_global_graph(".../global_features_graph.pkl", save_path="global.png")
    """
    artifact = GlobalGraphArtifact.load(global_graph_pkl)
    return artifact.draw_graph(**kwargs)


def draw_global_graph_time_lag(
    G: nx.DiGraph,
    var_names: Optional[Sequence[str]] = None,
    tau_max: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Global PCMCI Graph (Lag Layout)",
    # filtering
    p_threshold: float = 0.004,          # keep edges with p <= threshold
    use_p_fdr: bool = True,              # prefer p_fdr if present
    min_effect: float = 0.10,            # keep edges with weight >= min_effect (if weight exists)
    # layout styling
    x_gap: float = 4.0,
    y_gap: float = 1.0,
    node_size: int = 1700,
    font_size: int = 9,
    edge_width_scale: float = 4.0,
    arrow_size: int = 14,
    margin_pts: int = 16,               # prevents edges drawing over nodes
    curve_rad: float = 0.10,
    figsize=(16, 9),
):
    """
    Draw a GLOBAL PCMCI graph in left-to-right time-lag columns.

    Assumes:
      - Nodes in G are base variable names (e.g., "q10_ttc_proxy")
      - Edges have attributes like: tau (int), p / p_fdr (float), weight (float)

    Creates a layered graph with nodes formatted as:
      base(t-τ) in earlier columns and base(t) in the last column.
    """

    if G is None or G.number_of_edges() == 0:
        print("No edges to draw.")
        return

    # ---- Restrict variables if requested ----
    if var_names is not None:
        keep = set(var_names)
        base_nodes = [n for n in G.nodes() if str(n) in keep]
        if not base_nodes:
            print("No matching variables found in graph for var_names.")
            return
    else:
        base_nodes = list(G.nodes())

    # ---- Determine tau_max from edges if not given ----
    taus = []
    for u, v, d in G.edges(data=True):
        if var_names is not None and (str(u) not in keep or str(v) not in keep):
            continue
        if "tau" in d:
            try:
                taus.append(int(d["tau"]))
            except Exception:
                pass
    if tau_max is None:
        tau_max = max(taus) if taus else 1
    tau_max = max(1, int(tau_max))

    # ---- Build layered (time-expanded) graph ----
    # Node format matches your existing style: "name(t)" or "name(t-2)"
    def node_label(base: str, lag: int) -> str:
        return f"{base}(t)" if lag == 0 else f"{base}(t-{lag})"

    LG = nx.DiGraph()

    # Add nodes for all bases across lags that appear in kept edges
    # We only need (t) for targets and (t-τ) for sources.
    # But for neat columns, we include all bases at all lags 0..tau_max.
    bases = sorted({str(n) for n in base_nodes})
    for base in bases:
        for lag in range(tau_max, -1, -1):
            LG.add_node(node_label(base, lag))

    # Add edges: u(t-τ) -> v(t) with same attributes
    kept_edges = []
    for u, v, d in G.edges(data=True):
        su, sv = str(u), str(v)
        if var_names is not None and (su not in keep or sv not in keep):
            continue

        tau = int(d.get("tau", 1))
        if tau < 1 or tau > tau_max:
            continue

        # pick p-value field
        p_val = d.get("p_fdr") if (use_p_fdr and "p_fdr" in d) else d.get("p", 1.0)
        try:
            p_val = float(p_val)
        except Exception:
            p_val = 1.0

        w_val = d.get("weight", None)
        if w_val is not None:
            try:
                w_val = float(w_val)
            except Exception:
                w_val = None

        # filter by p-value and effect
        if p_val > p_threshold:
            continue
        if (w_val is not None) and (abs(w_val) < float(min_effect)):
            continue

        src = node_label(su, tau)
        tgt = node_label(sv, 0)

        LG.add_edge(src, tgt, **d)
        kept_edges.append((src, tgt, d))

    if LG.number_of_edges() == 0:
        print("No edges survived filtering. (Try increasing p_threshold or lowering min_effect.)")
        return

    # ---- Build positions: one column per lag, left->right: t-τmax ... t-1 ... t ----
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

    lag_to_nodes = {}
    all_base = set()

    for n in LG.nodes():
        base, lag = parse_node(n)
        if base is None:
            continue
        all_base.add(base)
        lag_to_nodes.setdefault(lag, []).append(n)

    # If var_names provided, restrict to those bases for consistent ordering
    if var_names is not None:
        base_sorted = [b for b in var_names if b in all_base]
    else:
        base_sorted = sorted(all_base)

    lags_sorted = list(range(tau_max, -1, -1))  # τmax ... 0

    pos = {}
    for col, lag in enumerate(lags_sorted):
        x = col * x_gap
        nodes = lag_to_nodes.get(lag, [])
        base_to_node = {}
        for n in nodes:
            b, _ = parse_node(n)
            base_to_node[b] = n

        for row, base in enumerate(base_sorted):
            if base in base_to_node:
                pos[base_to_node[base]] = (x, -row * y_gap)

    # fallback for any missing node positions (shouldn't happen)
    for i, n in enumerate(LG.nodes()):
        if n not in pos:
            pos[n] = (0.0, -(len(base_sorted) + 2 + i) * y_gap)

    # ---- Edge styling (match your p-value gating idea) ----
    edge_colors = []
    edge_widths = []
    edgelist = []

    for u, v, data in LG.edges(data=True):
        p_val = data.get("p_fdr") if (use_p_fdr and "p_fdr" in data) else data.get("p", 1.0)
        w = data.get("weight", 1.0)

        try:
            p_val = float(p_val)
        except Exception:
            p_val = 1.0
        try:
            w = float(w)
        except Exception:
            w = 1.0

        # color tiers similar to yours (you can adjust)
        if p_val < 1e-6:
            edge_colors.append("red")
        else:
            edge_colors.append("orange")

        edge_widths.append(max(0.5, edge_width_scale * abs(w)))
        edgelist.append((u, v))

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(LG, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_labels(LG, pos, ax=ax, font_size=font_size)

    # important: avoid drawing edges over nodes
    nx.draw_networkx_edges(
        LG, pos, ax=ax,
        edgelist=edgelist,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=arrow_size,
        min_source_margin=margin_pts,
        min_target_margin=margin_pts,
        connectionstyle=f"arc3,rad={curve_rad}",
    )

    ax.set_title(title)
    ax.set_axis_off()

    # annotate x-axis columns with lag headers
    for col, lag in enumerate(lags_sorted):
        x = col * x_gap
        header = "t" if lag == 0 else f"t-{lag}"
        ax.text(x, 1.0 * y_gap, header, ha="center", va="bottom", fontsize=font_size+2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
