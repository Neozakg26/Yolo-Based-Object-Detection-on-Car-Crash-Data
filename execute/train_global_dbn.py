# execute/train_global_dbn.py

import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from explainability.hierarchical_dbn import AccidentRiskAssessor


def parse_scene_id_from_track_path(track_path: str) -> str:
    # expects: .../000001_tracks.parquet
    base = os.path.basename(track_path)
    return base.split("_")[0]


def load_metadata_table(csv_path: str) -> pd.DataFrame:
    meta = pd.read_csv(csv_path)

    # normalize vidname to zero-padded 6 chars where possible
    meta["vidname"] = meta["vidname"].astype(str).str.zfill(6)

    # compute accident_start_frame_in_clip from frame_1..frame_50
    frame_cols = [c for c in meta.columns if c.startswith("frame_")]
    # ensure correct order frame_1..frame_50
    def frame_idx(c: str) -> int:
        return int(c.split("_")[1])
    frame_cols = sorted(frame_cols, key=frame_idx)

    def first_one(row) -> Optional[int]:
        for c in frame_cols:
            try:
                if int(row[c]) == 1:
                    return int(c.split("_")[1])
            except Exception:
                continue
        return None

    meta["accident_start_frame"] = meta.apply(first_one, axis=1)

    # normalize egoinvolve
    if "egoinvolve" in meta.columns:
        meta["egoinvolve"] = meta["egoinvolve"].astype(str).str.strip()
    else:
        meta["egoinvolve"] = "Unknown"

    return meta


def get_scene_meta(meta_df: pd.DataFrame, scene_id: str) -> Dict:
    row = meta_df[meta_df["vidname"] == scene_id]
    if row.empty:
        # still return something usable
        return {"vidname": scene_id, "accident_start_frame": None, "egoinvolve": "Unknown"}
    r = row.iloc[0].to_dict()
    # only keep fields we need downstream, but safe to pass more
    return {
        "vidname": scene_id,
        "accident_start_frame": r.get("accident_start_frame", None),
        "egoinvolve": r.get("egoinvolve", "Unknown"),
        "weather": r.get("weather", None),
        "timing": r.get("timing", None),
        "youtubeID": r.get("youtubeID", None),
    }


def load_scene_pair(results_dir: str, scene_id: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    track_path = os.path.join(results_dir, f"{scene_id}_tracks.parquet")
    env_path = os.path.join(results_dir, f"{scene_id}_env.parquet")

    tracks_df = pd.read_parquet(track_path)
    env_df = None
    if os.path.exists(env_path):
        env_df = pd.read_parquet(env_path)

    return tracks_df, env_df


def main():
    argparrser = argparse.ArgumentParser()
    argparrser.add_argument("--results_dir", type=str, required=True,
                    help="Folder containing *_tracks.parquet and *_env.parquet")
    argparrser.add_argument("--meta_csv", type=str, required=True,
                    help="Crash_Table.csv path")
    argparrser.add_argument("--features_path", type=str, required=True,
                    help="Pickle file containing GLOBAL PCMCI grargparrserh (nx.DiGrargparrserh or payload dict with key 'graph')")
    argparrser.add_argument("--out_model", type=str, required=True,
                    help="Output path for global DBN model parquet (e.g. global_risk_model.parquet)")

    # training options
    argparrser.add_argument("--preacc_only", action="store_true", default=True,
                    help="Use only frames strictly before accident_start_frame (default True).")
    argparrser.add_argument("--min_preacc_frames", type=int, default=10,
                    help="Skip scenes with fewer than this many pre-acc frames (prevents unstable training).")

    # model options
    argparrser.add_argument("--inference_method", type=str, default="supervised",
                    choices=["supervised", "belief_propagation", "variable_elimination"])
    argparrser.add_argument("--prior_strength", type=float, default=10.0)
    argparrser.add_argument("--video_fps", type=float, default=10.0)

    # classifier training
    argparrser.add_argument("--train_classifiers", action="store_true", default=True,
                    help="Train global risk + ego classifiers.")
    argparrser.add_argument("--random_state", type=int, default=42)

    args = argparrser.parse_args()

    results_dir = args.results_dir
    meta_df = load_metadata_table(args.meta_csv)

    # load global PCMCI graph
    with open(args.features_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "graph" in obj:
        pcmci_graph = obj["graph"]
    else:
        pcmci_graph = obj

    # enumerate scenes by track files
    track_files = sorted(glob.glob(os.path.join(results_dir, "*_tracks.parquet")))
    if not track_files:
        raise FileNotFoundError(f"No *_tracks.parquet found in {results_dir}")

    scene_items = []
    skipped = 0

    for tp in track_files:
        scene_id = parse_scene_id_from_track_path(tp)
        meta = get_scene_meta(meta_df, scene_id)

        tracks_df, env_df = load_scene_pair(results_dir, scene_id)

        # Merge env onto tracks in the assessor; but we still need truncation by frame
        # Determine accident frame in clip (1..50) — our tracks typically use 0-based frame indexes.
        acc_frame_clip = meta.get("accident_start_frame", None)
        if acc_frame_clip is not None and pd.notna(acc_frame_clip):
            # Convert 1..50 -> 0..49 if your tracks are 0-based.
            # If your tracks are already 1-based, remove the -1.
            acc_frame_local = int(acc_frame_clip) - 1
        else:
            acc_frame_local = None

        if args.preacc_only and acc_frame_local is not None:
            # strictly pre-accident frames
            tracks_df = tracks_df[tracks_df["frame"] < acc_frame_local].copy()
            if env_df is not None:
                env_df = env_df[env_df["frame"] < acc_frame_local].copy()

        if len(tracks_df["frame"].unique()) < args.min_preacc_frames:
            skipped += 1
            continue

        scene_items.append((scene_id, tracks_df, env_df, meta))

    if not scene_items:
        raise RuntimeError("No scenes available after filtering/truncation.")

    print(f"Scenes loaded: {len(scene_items)} (skipped: {skipped})")

    assessor = AccidentRiskAssessor(
        inference_method=args.inference_method,
        prior_strength=args.prior_strength,
        video_fps=args.video_fps,
    )

    print("Fitting GLOBAL DBN/CPT (and classifiers if enabled)...")
    assessor.fit_global(
        scene_items=scene_items,
        pcmci_graph=pcmci_graph,
        train_classifiers=args.train_classifiers,
        random_state=args.random_state,
    )

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assessor.save(out_path)

    print(f"\nGlobal model saved: {out_path}")
    clf_path = out_path.with_suffix(".classifier.pkl")
    if clf_path.exists():
        print(f"Global classifiers saved: {clf_path}")


if __name__ == "__main__":
    main()
