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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# Matplotlib 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


RISK_LABELS = ["Safe", "Elevated", "Critical"]

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


def tta_to_label(tta_seconds: float) -> str:
    if tta_seconds >= 2.5:
        return "Safe"
    elif tta_seconds >= 1.5:
        return "Elevated"
    else:
        return "Critical"

def make_frame_gt_labels(frames: np.ndarray, accident_frame_local: Optional[int], video_fps: float) -> Optional[pd.DataFrame]:
    """
    Build ground-truth labels per frame from accident start frame using TTA.
    frames: array of unique frame indices present (0-based)
    accident_frame_local: accident frame index (0-based) or None
    Returns DataFrame(frame, gt_label) or None if no accident frame.
    """
    if accident_frame_local is None:
        return None

    frames = np.asarray(sorted(frames))
    tta_frames = accident_frame_local - frames
    tta_seconds = tta_frames / float(video_fps)

    gt = [tta_to_label(float(t)) for t in tta_seconds]
    return pd.DataFrame({"frame": frames, "gt_label": gt})

def pred_probs_to_label(row: pd.Series) -> str:
    # expects columns P_Safe, P_Elevated, P_Critical
    probs = np.array([row.get("P_Safe", 0.0), row.get("P_Elevated", 0.0), row.get("P_Critical", 0.0)], dtype=float)
    return RISK_LABELS[int(np.argmax(probs))]

def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str, title: str = "Confusion Matrix"):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_example_trajectory(df_pred: pd.DataFrame, df_gt: Optional[pd.DataFrame],
                            save_path: str, title: str,
                            frames_dir: Optional[str] = None,
                            frame_glob: str ="*.jpg",
                            n_thumbs: int =5,
                            onset_frame: Optional[int] = None,
                            ):
    """
    df_pred: output from assessor.get_risk_trajectory -> frame, P_Safe, P_Elevated, P_Critical, risk_score, MAP_Risk
    df_gt: frame, gt_label (optional)
    """
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    ax.plot(df_pred["frame"], df_pred["P_Safe"], label="P(Safe)")
    ax.plot(df_pred["frame"], df_pred["P_Elevated"], label="P(Elevated)")
    ax.plot(df_pred["frame"], df_pred["P_Critical"], label="P(Critical)")
    ax.plot(df_pred["frame"], df_pred["risk_score"], label="risk_score")

    # ---------- determine GT onset frame ----------
    gt_onset = None
    if onset_frame is not None:
        gt_onset = int(onset_frame)
    elif df_gt is not None and "gt_label" in df_gt.columns:
        crit = df_gt[df_gt["gt_label"] == "Critical"]
        if len(crit) > 0:
            gt_onset = int(crit["frame"].iloc[0])

    # ---------- choose thumbnail frames (aligned with x-axis) ----------
    frames = df_pred["frame"].to_numpy()
    fmin, fmax = int(frames.min()), int(frames.max())

    # pick thumbnails: [start, midpoints..., end] and include onset neighborhood if available
    thumbs = np.linspace(fmin, fmax, num=max(n_thumbs, 2), dtype=int).tolist()

    if gt_onset is not None:
        # ensure onset and a couple neighbors are included (clipped to range)
        extra = [gt_onset - 3, gt_onset, gt_onset + 3]
        extra = [int(np.clip(x, fmin, fmax)) for x in extra]
        thumbs = list(dict.fromkeys(thumbs + extra))  # preserve order, unique

    thumbs = sorted(set(thumbs))
    # keep at most ~8 thumbnails to avoid clutter
    if len(thumbs) > 8:
        # keep extremes + onset + evenly-spaced
        keep = {thumbs[0], thumbs[-1]}
        if gt_onset is not None:
            keep.add(gt_onset)
        # fill remaining with evenly spaced
        remaining = [t for t in thumbs if t not in keep]
        k = max(0, 8 - len(keep))
        if k > 0 and len(remaining) > 0:
            idx = np.linspace(0, len(remaining) - 1, num=k, dtype=int)
            for i in idx:
                keep.add(remaining[i])
        thumbs = sorted(keep)

    # ---------- build figure layout ----------
    has_frames = frames_dir is not None and Path(frames_dir).exists()
    if has_frames:
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[3.2, 1.4], hspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
        ax_strip = fig.add_subplot(gs[1, 0])
    else:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax_strip = None

    # ---------- top plot: probabilities ----------
    ax.plot(df_pred["frame"], df_pred["P_Safe"], label="P(Safe)")
    ax.plot(df_pred["frame"], df_pred["P_Elevated"], label="P(Elevated)")
    ax.plot(df_pred["frame"], df_pred["P_Critical"], label="P(Critical)")
    ax.plot(df_pred["frame"], df_pred["risk_score"], label="risk_score")

    if gt_onset is not None:
        ax.axvline(gt_onset, linestyle="--", label="GT critical onset")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability / score")
    ax.legend(loc="best")

    # ---------- bottom plot: filmstrip ----------
    if ax_strip is not None:
        ax_strip.set_xlim(fmin, fmax)
        ax_strip.set_ylim(0, 1)
        ax_strip.axis("off")

        # Read images and place them as “data-aligned” thumbnails using extent
        # Each thumbnail spans a small x-range so it visually aligns with the timeline.
        span = max(1, int((fmax - fmin) / (len(thumbs) * 2)))  # width in "frame units"

        # Find candidate image files once
        # (Assumes frames are directly in frames_dir; if nested, pass the right frames_dir.)
        img_files = sorted(Path(frames_dir).glob(frame_glob))

        # Build a quick lookup from frame index -> filepath (handles common naming patterns)
        # If your names are e.g. frame_00023.jpg, 000023.jpg, img_00023.jpg, it will still work.
        def match_file_for_frame(k: int) -> Optional[Path]:
            k_strs = {
                str(k),
                f"{k:05d}",
                f"{k:06d}",
                f"frame_{k}",
                f"frame_{k:05d}",
                f"frame_{k:06d}",
                f"img_{k}",
                f"img_{k:05d}",
                f"img_{k:06d}",
            }
            for p in img_files:
                name = p.stem
                if any(s in name for s in k_strs):
                    return p
            # fallback: if files are strictly ordered and frame starts at 0 or 1
            # (common in extracted video frames)
            idx0 = k - fmin
            if 0 <= idx0 < len(img_files):
                return img_files[idx0]
            return None

        for k in thumbs:
            p = match_file_for_frame(k)
            if p is None or not p.exists():
                # if missing, just draw a placeholder tick label
                ax_strip.text(k, 0.5, f"{k}", ha="center", va="center", fontsize=8)
                continue

            try:
                img = mpimg.imread(str(p))
            except Exception:
                ax_strip.text(k, 0.5, f"{k}", ha="center", va="center", fontsize=8)
                continue

            # place image
            ax_strip.imshow(
                img,
                extent=(k - span, k + span, 0, 1),
                aspect="auto"
            )
            # add a small frame index label
            ax_strip.text(k, -0.05, f"{k}", ha="center", va="top", fontsize=8)

        # mark onset on strip too
        if gt_onset is not None:
            ax_strip.axvline(gt_onset, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

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
    argparrser.add_argument("--preacc_only", action="store_true", default=False,
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

    argparrser.add_argument("--test_split", type=float, default=0.2,
                    help="Fraction of scenes used for evaluation (default 0.2).")
    argparrser.add_argument("--plots_dir", type=str, default=None,
                    help="Where to save evaluation plots. Default: <out_model_dir>/eval_plots")
    argparrser.add_argument("--max_plot_scenes", type=int, default=5,
                    help="How many test scenes to plot trajectories for.")


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
    #Compute Scene Items 
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

    # --- 80/20 split by scene FROM SCENE_ITEMS ---
    rng = np.random.RandomState(args.random_state)
    idx = np.arange(len(scene_items))
    rng.shuffle(idx)

    n_test = max(1, int(round(len(scene_items) * float(args.test_split))))
    test_idx = set(idx[:n_test].tolist())

    train_items = [scene_items[i] for i in range(len(scene_items)) if i not in test_idx]
    test_items  = [scene_items[i] for i in range(len(scene_items)) if i in test_idx]

    print(f"Train scenes: {len(train_items)} | Test scenes: {len(test_items)} (split={1-float(args.test_split):.0%}/{float(args.test_split):.0%})")

    assessor = AccidentRiskAssessor(
        inference_method=args.inference_method,
        prior_strength=args.prior_strength,
        video_fps=args.video_fps,
    )

    print("Fitting GLOBAL DBN/CPT (and classifiers if enabled)...")
    assessor.fit_global(
        scene_items=train_items,
        pcmci_graph=pcmci_graph,
        train_classifiers=args.train_classifiers,
        random_state=args.random_state,
    )

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assessor.save(out_path)

    print(f"\n Global model saved: {out_path}")
    clf_path = out_path.with_suffix(".classifier.pkl")
    if clf_path.exists():
        print(f"Global classifiers saved: {clf_path}")

    # --- Evaluate on test scenes ---
    plots_dir = args.plots_dir
    if plots_dir is None:
        plots_dir = str(Path(args.out_model).parent / "eval_plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ------ Images directory  ( from results )
    images_dir = str(Path(args.results_dir).parent)
    

    y_true_all = []
    y_pred_all = []
    ego_true_all = []
    ego_pred_all = []

    plotted = 0

    for (scene_id, tracks_df, env_df, meta) in test_items:
        # IMPORTANT: for evaluation, use ORIGINAL local frame index, not the global offset used in fit_global.
        frames = np.asarray(sorted(tracks_df["frame"].unique()))

        acc_clip = meta.get("accident_start_frame", None)
        if acc_clip is not None and pd.notna(acc_clip):
            acc_local = int(acc_clip) - 1
        else:
            acc_local = None

        gt_df = make_frame_gt_labels(frames, acc_local, assessor.video_fps)

        pred_df = assessor.get_risk_trajectory(tracks_df, env_df)

        # ---------- EGO INVOLVE EVALUATION (scene-level) ----------
        # Ground truth ego-involve is scene-level in your metadata (Yes/No/Unknown)
        gt_ego = str(meta.get("egoinvolve", "Unknown")).strip()

        # Only evaluate if we have a usable binary GT label
        if gt_ego in {"Yes", "No"}:
            # Only evaluate if the predictor actually produced ego columns
            if "MAP_EgoInvolved" in pred_df.columns:
                # Scene-level prediction: majority vote across frames (more stable than per-frame)
                pred_scene_ego = (
                    pred_df["MAP_EgoInvolved"]
                    .astype(str)
                    .str.strip()
                    .replace({"YES": "Yes", "NO": "No", "yes": "Yes", "no": "No"})
                    .value_counts()
                    .idxmax()
                )

                ego_true_all.append(gt_ego)
                ego_pred_all.append(pred_scene_ego)

            # Optional fallback: if only probabilities exist but MAP is missing
            elif "P_EgoInvolved_Yes" in pred_df.columns and "P_EgoInvolved_No" in pred_df.columns:
                p_yes = float(pred_df["P_EgoInvolved_Yes"].mean())
                pred_scene_ego = "Yes" if p_yes >= 0.5 else "No"
                ego_true_all.append(gt_ego)
                ego_pred_all.append(pred_scene_ego)


        # Align + accumulate labels
        if gt_df is not None:
            merged = pred_df.merge(gt_df, on="frame", how="inner")
            if len(merged) > 0:
                merged["pred_label"] = merged.apply(pred_probs_to_label, axis=1)
                y_true_all.extend(merged["gt_label"].tolist())
                y_pred_all.extend(merged["pred_label"].tolist())

        # Plot a few example trajectories
        if plotted < args.max_plot_scenes:
            plot_path = str(Path(plots_dir) / f"{scene_id}_risk_trajectory.png")

            plot_example_trajectory(pred_df,
                                    gt_df,
                                    plot_path, 
                                    f"Risk Trajectory - Scene {scene_id}",
                                    frames_dir=images_dir,
                                    frame_glob=f"C_{scene_id}_*.jpg")
            plotted += 1

    # Metrics
    if len(y_true_all) == 0:
        print("No GT labels available for evaluation (missing accident_start_frame or no overlapping frames).")
    else:
        acc = accuracy_score(y_true_all, y_pred_all)
        f1m = f1_score(y_true_all, y_pred_all, labels=RISK_LABELS, average="macro")
        cm = confusion_matrix(y_true_all, y_pred_all, labels=RISK_LABELS)

        print("\n=== Held-out Test Evaluation (scene-level 80/20 split) ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro-F1:  {f1m:.3f}")
        print("\nClassification report:")
        print(classification_report(y_true_all, y_pred_all, labels=RISK_LABELS))

        cm_path = str(Path(plots_dir) / "confusion_matrix.png")
        plot_confusion_matrix(cm, RISK_LABELS, cm_path, "Confusion Matrix (Test Scenes)")

        metrics_path = str(Path(plots_dir) / "test_metrics.csv")
        pd.DataFrame([{
            "accuracy": acc,
            "macro_f1": f1m,
            "n_frames_eval": len(y_true_all),
            "n_test_scenes": len(test_items),
            "test_split": float(args.test_split),
            "random_state": int(args.random_state),
            "preacc_only": bool(args.preacc_only),
        }]).to_csv(metrics_path, index=False)

        print(f"\nSaved evaluation outputs to: {plots_dir}")

if __name__ == "__main__":
    main()
