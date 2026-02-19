# execute/train_global_dbn.py

import argparse
import glob
import os
import pickle
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from explainability.hierarchical_dbn import AccidentRiskAssessor
# SkLearn 
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# Matplotlib 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


RISK_LABELS = ["Safe", "Elevated", "Critical"]
logger = logging.getLogger(__name__)
logging.basicConfig(filename="train_global_dbn.log",level=logging.INFO)

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

def compute_scene_items(track_files,meta_df,results_dir):
    scene_items = []
    skipped =0
    for tp in track_files:
        scene_id = parse_scene_id_from_track_path(tp)
        meta = get_scene_meta(meta_df, scene_id)

        tracks_df, env_df = load_scene_pair(results_dir, scene_id)

        # Merge env onto tracks in the assessor; but we still need truncation by frame
        # Determine accident frame in clip (1..50) — our tracks typically use 0-based frame indexes.
        acc_frame_clip = meta.get("accident_start_frame", None)
        if acc_frame_clip is not None and pd.notna(acc_frame_clip):
            # Convert 1..50 -> 0..49 if your tracks are 0-based.
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
    return scene_items, skipped

def scene_strat_label(meta: Dict) -> int:
    """
    Returns 1 for accident scenes, 0 otherwise.
    """
    acc = meta.get("accident_start_frame", None)
    return int(acc is not None and pd.notna(acc))


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

def plot_example_trajectory(
    df_pred: pd.DataFrame,
    df_gt: Optional[pd.DataFrame],
    save_path: str,
    title: str,
    frames_dir: Optional[str] = None,
    frame_glob: str = "*.jpg",
    onset_frame: Optional[int] = None,
    scene_id: str= "",
    thumb_cols: int = 8,          # bigger thumbs -> fewer columns
    thumb_scale: float = 1.35,    # increases overall thumbnail area

):
    """
    Plots risk trajectory + ALL thumbnails up to accident onset (or all frames if no onset).

    frames_dir: directory containing frames for this scene
    frame_glob: pattern to match scene frames within frames_dir (e.g., f"C_{scene_id}_*.jpg")
    thumb_cols: how many thumbnails per row (smaller => bigger thumbs)
    thumb_scale: scales the figure height allocated to thumbnails
    """
   # ---------- determine GT onset frame ----------
    gt_onset = None
    if onset_frame is not None:
        gt_onset = int(onset_frame)
    elif df_gt is not None and "gt_label" in df_gt.columns:
        crit = df_gt[df_gt["gt_label"] == "Critical"]
        if len(crit) > 0:
            gt_onset = int(crit["frame"].iloc[0])

    frames = df_pred["frame"].to_numpy()
    fmin, fmax = int(frames.min()), int(frames.max())

    # thumbnails: every frame up to onset (inclusive). If no onset -> all frames.
   
    f_end = min(fmax+4,50) 

    thumbs = list(range(fmin, f_end + 1))
    n_thumbs = len(thumbs)

    # ---------- collect image files ----------
    has_frames = frames_dir is not None and Path(frames_dir).exists()
    logger.info(f"Has Frames boolean is {has_frames} IN {frames_dir}")
    img_files = []
    if has_frames:
        img_files = sorted(Path(frames_dir).glob(frame_glob))
        if len(img_files) == 0:
            has_frames = False  # fallback: just plot curves

    # ---------- figure layout ----------
    # Make the figure taller as thumbnails increase (but keep it reasonable)
    # For CCD (<=50 frames), this is fine.
    ncols = max(1, int(thumb_cols))
    nrows = int(math.ceil(n_thumbs / ncols)) if has_frames else 0

    # height: top plot ~4, each thumb row adds ~1.2 (scaled)
    fig_h = 4.0 + (nrows * 1.25 * thumb_scale if has_frames else 0.0)
    fig_w = 12.0

    if has_frames:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(2, 1, height_ratios=[3.2, max(1.0, 1.2 * nrows * thumb_scale)], hspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
        gs_thumbs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[1, 0], wspace=0.02, hspace=0.05)
    else:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        gs_thumbs = None

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

    # ---------- helper: robust frame->image mapping ----------
    # We try matching by frame index in filename; otherwise map by position.
    def match_file_for_frame(scene_id: str,k : int) -> Optional[Path]:
        if not img_files:
            return None

        # try explicit name match
        k_strs = {
            str(k),
            f"C_{scene_id}_{k:02d}",
        }
        for p in img_files:
            name = p.stem
            if any(s in name for s in k_strs):
                logger.info(f"File Found in match_file_for_frame  C_{scene_id}_{k:02d}")
                return p
        logger.info(f"file not in match_file_for_frame  C_{scene_id}_{k:02d}")
        # robust fallback: map frame position into file index
        # handles missing/shifted numbering (0/1-based), truncation, gaps, etc.
        if f_end > fmin:
            idx0 = int(round((k - fmin) / (f_end - fmin) * (len(img_files) - 1)))
        else:
            idx0 = 0
        idx0 = int(np.clip(idx0, 0, len(img_files) - 1))
        return img_files[idx0]
    # ---------- bottom: thumbnails grid ----------
    if has_frames and gs_thumbs is not None:
        for idx, k in enumerate(thumbs):
            r = idx // ncols
            c = idx % ncols
            ax_img = fig.add_subplot(gs_thumbs[r, c])
            ax_img.axis("off")
            logger.info(f"CALLING match_file_for_frame")
            p = match_file_for_frame(scene_id,k)
            if p is None or not p.exists():
                ax_img.text(0.5, 0.5, f"{k}", ha="center", va="center", fontsize=9)
                continue

            try:
                img = mpimg.imread(str(p))
                ax_img.imshow(img)
                # label frame number (small but readable)
                ax_img.set_title(f"f={k}", fontsize=9, pad=2)
            except Exception:
                ax_img.text(0.5, 0.5, f"{k}", ha="center", va="center", fontsize=9)
    else:
        logger.info(f"Failed on running match_file_for_frame: gs_thumbs: {gs_thumbs} and has_frames{has_frames}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main(args):

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


    #Compute Scene Items 
    scene_items, skipped  = compute_scene_items(track_files=track_files,
                                                     meta_df=meta_df,
                                                     results_dir=results_dir)
    
    if not scene_items:
        raise RuntimeError("No scenes available after filtering/truncation.")
 
    logger.info(f"Scenes loaded: {len(scene_items)} (skipped: {skipped})")

    # ---------------- 5-FOLD CROSS VALIDATION (by scene) ----------------
    n_splits = int(getattr(args, "n_splits", 5))
    if n_splits < 2:
        raise ValueError("--n_splits must be >= 2")

    # Stratify by whether the scene is accident-labeled or not
    y_scene = np.array([scene_strat_label(meta) for (_, _, _, meta) in scene_items], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    # Base plots dir
    plots_dir = args.plots_dir
    if plots_dir is None:
        plots_dir = str(Path(args.out_model).parent / "eval_plots_cv")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Images directory (from results)
    # images_dir = str(Path(args.results_dir).parent)

    images_dir = "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images"
    # Collect fold metrics
    fold_rows = []
    # Aggregate predictions across all folds
    y_true_all_cv = []
    y_pred_all_cv = []
    ego_true_all_cv = []
    ego_pred_all_cv = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(scene_items)), y_scene), start=1):
        fold_dir = Path(plots_dir) / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_items = [scene_items[i] for i in train_idx]
        test_items  = [scene_items[i] for i in test_idx]

        logger.info(f"[CV] Fold {fold_idx}/{n_splits} | Train scenes: {len(train_items)} | Test scenes: {len(test_items)}")

        assessor = AccidentRiskAssessor(
            inference_method=args.inference_method,
            prior_strength=args.prior_strength,
            video_fps=args.video_fps,
        )

        logger.info(f"[CV] Fold {fold_idx}: fitting GLOBAL DBN/CPT (and classifiers if enabled)...")
        assessor.fit_global(
            scene_items=train_items,
            pcmci_graph=pcmci_graph,
            train_classifiers=args.train_classifiers,
            random_state=args.random_state,
        )

        # (Optional) save model per fold if you want
        if getattr(args, "save_fold_models", False):
            fold_model_path = fold_dir / f"global_model_fold_{fold_idx:02d}.parquet"
            assessor.save(fold_model_path)
            logger.info(f"[CV] Fold {fold_idx}: model saved to {fold_model_path}")

        # --- Evaluate on this fold's test scenes ---
        y_true_all = []
        y_pred_all = []
        ego_true_all = []
        ego_pred_all = []

        plotted = 0
        max_plot = int(args.max_plot_scenes)

        for (scene_id, tracks_df, env_df, meta) in test_items:
            frames = np.asarray(sorted(tracks_df["frame"].unique()))

            acc_clip = meta.get("accident_start_frame", None)
            if acc_clip is not None and pd.notna(acc_clip):
                acc_local = int(acc_clip) - 1
            else:
                acc_local = None

            g_truth_df = make_frame_gt_labels(frames, acc_local, assessor.video_fps)
            pred_df = assessor.get_risk_trajectory(tracks_df, env_df)

            # ---------- EGO INVOLVE EVALUATION (scene-level) ----------
            gt_ego = str(meta.get("egoinvolve", "Unknown")).strip()
            if gt_ego in {"Yes", "No"}:
                if "MAP_EgoInvolved" in pred_df.columns:
                    pred_scene_ego = (
                        pred_df["MAP_EgoInvolved"]
                        .astype(str).str.strip()
                        .replace({"YES": "Yes", "NO": "No", "yes": "Yes", "no": "No"})
                        .value_counts().idxmax()
                    )
                    ego_true_all.append(gt_ego)
                    ego_pred_all.append(pred_scene_ego)
                elif "P_EgoInvolved_Yes" in pred_df.columns and "P_EgoInvolved_No" in pred_df.columns:
                    p_yes = float(pred_df["P_EgoInvolved_Yes"].mean())
                    ego_true_all.append(gt_ego)
                    ego_pred_all.append("Yes" if p_yes >= 0.5 else "No")

            # ---------- Risk label evaluation (frame-level, only if accident frame exists) ----------
            if g_truth_df is not None:
                merged = pred_df.merge(g_truth_df, on="frame", how="inner")
                if len(merged) > 0:
                    merged["pred_label"] = merged.apply(pred_probs_to_label, axis=1)
                    y_true_all.extend(merged["gt_label"].tolist())
                    y_pred_all.extend(merged["pred_label"].tolist())

            # ---------- Plot a few example trajectories for this fold ----------
            if plotted < max_plot:
                plot_path = str(fold_dir / f"{scene_id}_risk_trajectory.png")
                logger.info(f"Plotting for {scene_id}")
                plot_example_trajectory(
                    pred_df,
                    g_truth_df,
                    plot_path,
                    f"Risk Trajectory - Scene {scene_id} (Fold {fold_idx})",
                    frames_dir=images_dir,
                    frame_glob=f"C_{scene_id}_*.jpg",
                    scene_id=scene_id
                )
                plotted += 1

        # ---- Fold metrics (risk) ----
        if len(y_true_all) > 0:
            acc = accuracy_score(y_true_all, y_pred_all)
            f1m = f1_score(y_true_all, y_pred_all, labels=RISK_LABELS, average="macro")
            cm = confusion_matrix(y_true_all, y_pred_all, labels=RISK_LABELS)

            logger.info(f"[CV] Fold {fold_idx} risk accuracy: {acc:.3f}, macro-f1: {f1m:.3f}")
            logger.info("\n" + classification_report(y_true_all, y_pred_all, labels=RISK_LABELS))

            cm_path = str(fold_dir / "confusion_matrix.png")
            plot_confusion_matrix(cm, RISK_LABELS, cm_path, f"Confusion Matrix (Fold {fold_idx})")
        else:
            acc, f1m = np.nan, np.nan
            logger.info(f"[CV] Fold {fold_idx}: no GT risk labels available in this fold (no accident_start_frame overlap).")

        # ---- Fold metrics (ego) ----
        if len(ego_true_all) > 0:
            ego_acc = accuracy_score(ego_true_all, ego_pred_all)
            ego_f1 = f1_score(ego_true_all, ego_pred_all, labels=["No", "Yes"], average="macro")
            logger.info(f"[CV] Fold {fold_idx} ego accuracy: {ego_acc:.3f}, macro-f1: {ego_f1:.3f}")
        else:
            ego_acc, ego_f1 = np.nan, np.nan
            logger.info(f"[CV] Fold {fold_idx}: no usable ego labels found (expected 'Yes'/'No') or ego columns missing.")

        # Save fold metrics CSV
        fold_metrics_path = fold_dir / "fold_metrics.csv"
        pd.DataFrame([{
            "fold": fold_idx,
            "risk_accuracy": acc,
            "risk_macro_f1": f1m,
            "ego_accuracy": ego_acc,
            "ego_macro_f1": ego_f1,
            "n_frames_eval": len(y_true_all),
            "n_scenes_test": len(test_items),
            "random_state": int(args.random_state),
            "preacc_only": bool(args.preacc_only),
        }]).to_csv(fold_metrics_path, index=False)

        # accumulate across folds
        fold_rows.append({
            "fold": fold_idx,
            "risk_accuracy": acc,
            "risk_macro_f1": f1m,
            "ego_accuracy": ego_acc,
            "ego_macro_f1": ego_f1,
            "n_frames_eval": len(y_true_all),
            "n_scenes_test": len(test_items),
        })

        y_true_all_cv.extend(y_true_all)
        y_pred_all_cv.extend(y_pred_all)
        ego_true_all_cv.extend(ego_true_all)
        ego_pred_all_cv.extend(ego_pred_all)

    # ---------------- CV SUMMARY ----------------
    summary_df = pd.DataFrame(fold_rows)
    summary_path = Path(plots_dir) / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    def mean_std(series: pd.Series) -> str:
        s = series.dropna()
        if len(s) == 0:
            return "n/a"
        return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"

    logger.info("\n=== 5-Fold CV Summary ===")
    logger.info(f"Risk Accuracy: {mean_std(summary_df['risk_accuracy'])}")
    logger.info(f"Risk Macro-F1:  {mean_std(summary_df['risk_macro_f1'])}")
    logger.info(f"Ego Accuracy:  {mean_std(summary_df['ego_accuracy'])}")
    logger.info(f"Ego Macro-F1:   {mean_std(summary_df['ego_macro_f1'])}")
    logger.info(f"Saved CV summary to: {summary_path}")

    # Optional: overall confusion matrix across all folds (risk)
    if len(y_true_all_cv) > 0:
        cm_all = confusion_matrix(y_true_all_cv, y_pred_all_cv, labels=RISK_LABELS)
        cm_all_path = str(Path(plots_dir) / "confusion_matrix_all_folds.png")
        plot_confusion_matrix(cm_all, RISK_LABELS, cm_all_path, "Confusion Matrix (All Folds)")


if __name__ == "__main__":
    #Accept Arguments  
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
    argparrser.add_argument("--n_splits", type=int, default=5,
                    help="Number of CV folds (default 5).")
    argparrser.add_argument("--save_fold_models", action="store_true", default=False,
                    help="If set, save a model parquet for each fold inside the fold folder.")


    args = argparrser.parse_args()
  
    main(args)
