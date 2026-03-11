import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from explainability.hierarchical_dbn import AccidentRiskAssessor



def parse_scene_id_from_track_path(track_path: str) -> str:
    base = os.path.basename(track_path)
   # base.replace("tracks.parquet", "")
    return base.replace("_tracks.parquet", "")  #expects 000001_tracks.parquet


def load_scene_pair(results_dir: str, scene_id: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    track_path = os.path.join(results_dir, f"{scene_id}_tracks.parquet")
    env_path = os.path.join(results_dir, f"{scene_id}_env.parquet")

    tracks_df = pd.read_parquet(track_path)
    env_df = pd.read_parquet(env_path) if os.path.exists(env_path) else None
    return tracks_df, env_df


def scene_pred_any_above(pred_df: pd.DataFrame, thr: float = 0.7) -> int:
    """1 = accident, 0 = normal"""
    if len(pred_df) == 0 or "risk_score" not in pred_df.columns:
        return 0
    return int((pred_df["risk_score"].to_numpy() >= thr).any())


def scene_pred_k_consecutive(pred_df: pd.DataFrame, thr: float = 0.7, k: int = 2) -> int:
    """1 = accident, 0 = normal; requires k consecutive frames above thr."""
    if len(pred_df) == 0 or "risk_score" not in pred_df.columns:
        return 0
    x = (pred_df["risk_score"].to_numpy() >= thr).astype(int)
    if k <= 1:
        return int(x.any())
    # count longest run of 1s
    run = 0
    best = 0
    for v in x:
        run = run + 1 if v == 1 else 0
        best = max(best, run)
    return int(best >= k)


def plot_confusion_matrix(cm: np.ndarray, labels, save_path: str, normalize: bool = True, title: str = ""):
    if normalize:
        cm_plot = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fmt = ".2f"
        title = title or "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = title or "Confusion Matrix"

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(j, i, format(cm_plot[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Folder containing *_tracks.parquet and *_env.parquet")
    ap.add_argument("--model_path", type=str, required=True, help="Saved AccidentRiskAssessor model (parquet or pkl alias)")
    ap.add_argument("--scene_labels", type=str, required=True,
                    help="CSV with columns: scene_id,label where label is 1 for accident, 0 for normal.")
    ap.add_argument("--thr", type=float, default=0.70, help="risk_score threshold")
    ap.add_argument("--k", type=int, default=2, help="consecutive frames required (k>=1)")
    ap.add_argument("--out_dir", type=str, default="scene_eval_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading global model.")
    assessor = AccidentRiskAssessor.load(args.model_path)
    #assessor.inference_method = "supervised"  # To override the inference
    print("Loaded global model.")

    # Load labels
    lab = pd.read_csv(args.scene_labels)
    # normalize scene_id to zero-padded if needed

    label_map: Dict[str, int] = dict(zip(lab["scene_id"], lab["label"].astype(int)))

    # enumerate scenes
    track_files = sorted(glob.glob(os.path.join(args.results_dir, "*tracks.parquet")))
    if not track_files:
        raise FileNotFoundError(f"No *_tracks.parquet found in {args.results_dir}")

    y_true = []
    y_pred_any = []
    y_pred_k = []
    rows = []

    print(f"Track files \n {track_files}")
    for tp in track_files:
        scene_id = parse_scene_id_from_track_path(tp)
        if scene_id not in label_map:
            print(f"Skipping scene {scene_id}")
            continue  # skip if not labeled
        print(f"Processing scene {scene_id}")
        gt = int(label_map[scene_id])
        tracks_df, env_df = load_scene_pair(args.results_dir, scene_id)

        pred_df = assessor.get_risk_trajectory(tracks_df, env_df)

        pred_any = scene_pred_any_above(pred_df, thr=args.thr)
        pred_k = scene_pred_k_consecutive(pred_df, thr=args.thr, k=args.k)

        y_true.append(gt)
        y_pred_any.append(pred_any)
        y_pred_k.append(pred_k)

        mx = float(pred_df["risk_score"].max()) if len(pred_df) and "risk_score" in pred_df else float("nan")
        rows.append({
            "scene_id": scene_id,
            "gt": gt,
            "pred_any": pred_any,
            "pred_k": pred_k,
            "max_risk_score": mx,
            "thr": args.thr,
            "k": args.k,
        })

    if len(y_true) == 0:
        raise RuntimeError("No labeled scenes were evaluated. Check scene_labels_csv matches scene ids in results_dir.")

    # Metrics
    def summarize(name, yt, yp):
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        print(f"\n=== {name} ===")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1:        {f1:.3f}")
        print("\nReport:")
        print(classification_report(yt, yp, target_names=["Normal", "Accident"], zero_division=0))
        return acc, prec, rec, f1, cm

    acc1, p1, r1, f11, cm1 = summarize(f"Rule: ANY frame risk_score >= {args.thr}", y_true, y_pred_any)
    acc2, p2, r2, f12, cm2 = summarize(f"Rule: {args.k} consecutive frames risk_score >= thr", y_true, y_pred_k)

    # Save outputs
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "scene_predictions.csv", index=False)

    plot_confusion_matrix(cm1, ["Normal", "Accident"], str(out_dir / "cm_any_norm.png"),
                          normalize=True, title="Normalized CM: ANY>=thr")
    plot_confusion_matrix(cm2, ["Normal", "Accident"], str(out_dir / "cm_k_norm.png"),
                          normalize=True, title=f"Normalized CM: K={args.k}>=thr")

    pd.DataFrame([{
        "rule": "any",
        "thr": args.thr,
        "k": 1,
        "accuracy": acc1, "precision": p1, "recall": r1, "f1": f11,
        "n_scenes": len(y_true),
    }, {
        "rule": "k_consecutive",
        "thr": args.thr,
        "k": args.k,
        "accuracy": acc2, "precision": p2, "recall": r2, "f1": f12,
        "n_scenes": len(y_true),
    }]).to_csv(out_dir / "scene_metrics.csv", index=False)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()