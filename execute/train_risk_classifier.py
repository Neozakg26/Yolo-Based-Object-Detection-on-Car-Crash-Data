"""
Train GradientBoostingClassifier and StandardScaler on multiple scenes.

Trains two classifiers:
1. Risk classifier: Predicts Safe/Elevated/Critical based on TTA
2. Ego-involved classifier: Predicts whether ego vehicle is involved in accident
 
Usage:
    python execute/train_risk_classifier.py \
        --results_dir car_crash_dataset/CCD_images/results \
        --meta_path car_crash_dataset/CCD_images/Crash_Table.csv \
        --output_path models/risk_classifier.pkl
"""

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

from explainability.hierarchical_dbn.discretizer import ObservableDiscretizer


def get_scene_metadata(meta_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract accident_start_frame and ego_involved for all scenes from Crash_Table.csv.

    Returns:
        Dict mapping scene_id -> {"accident_frame": int, "ego_involved": bool}
    """
    df_meta = pd.read_csv(meta_path, dtype={"vidname": "string", "egoinvolve": "string"})

    frame_cols = [c for c in df_meta.columns if c.startswith("frame_")]

    df_long = df_meta.melt(
        id_vars=["vidname", "egoinvolve"],
        value_vars=frame_cols,
        var_name="frame",
        value_name="accident_flag"
    )

    df_long["frame"] = (
        df_long["frame"]
        .str.replace("frame_", "", regex=False)
        .astype("int32")
    )

    # Get first accident frame per scene
    accident_starts = (
        df_long[df_long["accident_flag"] == 1]
        .groupby("vidname")["frame"]
        .min()
        .to_dict()
    )

    # Get ego_involved per scene (same for all frames in a scene)
    ego_involved = (
        df_meta.set_index("vidname")["egoinvolve"]
        .apply(lambda x: 1 if x == "Yes" else 0)
        .to_dict()
    )

    # Combine into single dict
    scene_metadata = {}
    for scene_id in accident_starts.keys():
        scene_metadata[scene_id] = {
            "accident_frame": accident_starts[scene_id],
            "ego_involved": ego_involved.get(scene_id, 0)
        }


    return scene_metadata


def visualize_results(
    risk_classifier: GradientBoostingClassifier,
    ego_classifier: GradientBoostingClassifier,
    risk_metrics: Dict[str, Any],
    ego_metrics: Dict[str, Any],
    feature_cols: List[str],
    output_path: Path = None,
    show_plots: bool = True,
) -> None:
    """
    Visualize validation results with matplotlib.

    Creates a figure with:
    - Risk classifier confusion matrix
    - Ego classifier confusion matrix
    - Feature importance (top 15)
    - Cross-validation scores comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Model Validation Results", fontsize=14, fontweight='bold')

    # 1. Risk Classifier Confusion Matrix
    ax1 = axes[0, 0]
    cm_risk = confusion_matrix(risk_metrics['y_test'], risk_metrics['y_test_pred'])
    disp_risk = ConfusionMatrixDisplay(
        confusion_matrix=cm_risk,
        display_labels=['Safe', 'Elevated', 'Critical']
    )
    disp_risk.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title(f"Risk Classifier - Test Confusion Matrix\n"
                  f"Accuracy: {risk_metrics['test_accuracy']:.1%}")

    # 2. Ego Classifier Confusion Matrix (Scene-Level)
    ax2 = axes[0, 1]
    cm_ego = confusion_matrix(ego_metrics['y_scene_true'], ego_metrics['y_scene_pred'])
    disp_ego = ConfusionMatrixDisplay(
        confusion_matrix=cm_ego,
        display_labels=['Not-Involved', 'Involved']
    )
    disp_ego.plot(ax=ax2, cmap='Greens', colorbar=False)
    ax2.set_title(f"Ego-Involved Classifier - Test Confusion Matrix (Scene-Level)\n"
                  f"Accuracy: {ego_metrics['test_accuracy_scene_level']:.1%}")

    # 3. Feature Importance (Top 15)
    ax3 = axes[1, 0]
    importance = risk_classifier.feature_importances_
    indices = np.argsort(importance)[::-1][:15]  # Top 15
    top_features = [feature_cols[i] for i in indices]
    top_importance = importance[indices]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax3.barh(range(len(top_features)), top_importance, color=colors)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features)
    ax3.invert_yaxis()
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Risk Classifier - Top 15 Feature Importances')

    # Add value labels on bars
    for bar, val in zip(bars, top_importance):
        ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    # 4. Cross-Validation Scores Comparison
    ax4 = axes[1, 1]

    # Prepare data for grouped bar chart
    models = ['Risk Classifier', 'Ego Classifier']
    cv_means = [risk_metrics['cv_mean'], ego_metrics['cv_mean']]
    cv_stds = [risk_metrics['cv_std'], ego_metrics['cv_std']]
    test_accs = [risk_metrics['test_accuracy'], ego_metrics['test_accuracy_scene_level']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax4.bar(x - width/2, cv_means, width, label='CV Accuracy',
                    yerr=[s*2 for s in cv_stds], capsize=5, color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, test_accs, width, label='Test Accuracy',
                    color='darkorange', alpha=0.8)

    ax4.set_ylabel('Accuracy')
    ax4.set_title('Cross-Validation vs Test Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(loc='lower right')
    ax4.set_ylim(0, 1.0)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        fig_path = output_path.parent / f"{output_path.stem}_validation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nValidation plot saved: {fig_path}")

    if show_plots:
        plt.show()

    return fig


def visualize_class_distributions(
    risk_metrics: Dict[str, Any],
    train_scenes: int,
    val_scenes: int,
    test_scenes: int,
    output_path: Path = None,
    show_plots: bool = True,
) -> None:
    """
    Visualize class distributions for train, validation, and test sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Data Distribution Analysis", fontsize=14, fontweight='bold')

    # 1. Risk Label Distribution (Train vs Val vs Test)
    ax1 = axes[0]
    labels = ['Safe', 'Elevated', 'Critical']
    train_dist = [
        risk_metrics['train_label_dist']['safe'],
        risk_metrics['train_label_dist']['elevated'],
        risk_metrics['train_label_dist']['critical']
    ]
    val_dist = [
        risk_metrics['val_label_dist']['safe'],
        risk_metrics['val_label_dist']['elevated'],
        risk_metrics['val_label_dist']['critical']
    ]
    test_dist = [
        risk_metrics['test_label_dist']['safe'],
        risk_metrics['test_label_dist']['elevated'],
        risk_metrics['test_label_dist']['critical']
    ]

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax1.bar(x - width, train_dist, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x, val_dist, width, label='Validation', color='forestgreen', alpha=0.8)
    bars3 = ax1.bar(x + width, test_dist, width, label='Test', color='darkorange', alpha=0.8)

    ax1.set_ylabel('Number of Frames')
    ax1.set_title('Risk Label Distribution (Train/Val/Test)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # 2. Train/Val/Test Split Summary (Pie chart)
    ax2 = axes[1]
    sizes = [train_scenes, val_scenes, test_scenes]
    labels_pie = [
        f'Train\n({train_scenes} scenes)',
        f'Validation\n({val_scenes} scenes)',
        f'Test\n({test_scenes} scenes)'
    ]
    colors = ['steelblue', 'forestgreen', 'darkorange']
    explode = (0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels_pie, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}
    )
    ax2.set_title('Train/Validation/Test Scene Split')

    plt.tight_layout()

    if output_path:
        fig_path = output_path.parent / f"{output_path.stem}_distributions.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved: {fig_path}")

    if show_plots:
        plt.show()

    return fig


def split_scenes_by_stratification(
    scene_ids: List[str],
    scene_metadata: Dict[str, Dict[str, Any]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split scenes into train/validation/test sets, stratified by ego_involved.

    Splitting at scene level prevents temporal data leakage since
    frames within a scene are temporally correlated.

    ML Standard Split:
    - Train: Used for model fitting
    - Validation: Used for hyperparameter tuning and early stopping
    - Test: Held out for final unbiased evaluation

    Args:
        scene_ids: List of scene IDs to split
        scene_metadata: Dict mapping scene_id -> {"accident_frame": int, "ego_involved": bool}
        val_size: Fraction of data for validation (default: 0.15)
        test_size: Fraction of data for testing (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_scene_ids, val_scene_ids, test_scene_ids)
    """
    # Filter to scenes that have metadata
    valid_scenes = np.array([s for s in scene_ids if s in scene_metadata])

    if len(valid_scenes) == 0:
        return [], [], []

    # Get stratification labels (ego_involved: positive=1, negative=0)
    stratify_labels = np.array([scene_metadata[s]["ego_involved"] for s in valid_scenes])

    # Print class distribution
    n_positive = (stratify_labels == 1).sum()
    n_negative = (stratify_labels == 0).sum()
    print(f"  Scene distribution: Positive (ego_involved)={n_positive}, Negative={n_negative}")

    # First split: separate test set
    # Use sklearn's train_test_split for proper stratification
    train_val_scenes, test_scenes, train_val_labels, _ = train_test_split(
        valid_scenes,
        stratify_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)

    train_scenes, val_scenes, _, _ = train_test_split(
        train_val_scenes,
        train_val_labels,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_labels
    )

    return list(train_scenes), list(val_scenes), list(test_scenes)


def print_split_statistics(
    train_scenes: List[str],
    val_scenes: List[str],
    test_scenes: List[str],
    scene_metadata: Dict[str, Dict[str, Any]]
) -> None:
    """Print balanced split statistics."""
    def get_stats(scenes):
        if not scenes:
            return 0, 0
        pos = sum(1 for s in scenes if scene_metadata.get(s, {}).get("ego_involved", 0) == 1)
        neg = len(scenes) - pos
        return pos, neg

    train_pos, train_neg = get_stats(train_scenes)
    val_pos, val_neg = get_stats(val_scenes)
    test_pos, test_neg = get_stats(test_scenes)

    total = len(train_scenes) + len(val_scenes) + len(test_scenes)

    print(f"\n  Split Statistics (Total: {total} scenes):")
    print(f"  {'Set':<12} {'Scenes':>8} {'Positive':>10} {'Negative':>10} {'Pos %':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Train':<12} {len(train_scenes):>8} {train_pos:>10} {train_neg:>10} {train_pos/len(train_scenes)*100 if train_scenes else 0:>7.1f}%")
    print(f"  {'Validation':<12} {len(val_scenes):>8} {val_pos:>10} {val_neg:>10} {val_pos/len(val_scenes)*100 if val_scenes else 0:>7.1f}%")
    print(f"  {'Test':<12} {len(test_scenes):>8} {test_pos:>10} {test_neg:>10} {test_pos/len(test_scenes)*100 if test_scenes else 0:>7.1f}%")


def discover_scenes(results_dir: Path) -> List[str]:
    """
    Discover all scene IDs from tracks parquet files in results directory.

    Returns:
        List of scene IDs (e.g., ["000001", "000002", ...])
    """
    track_files = list(results_dir.glob("*_tracks.parquet"))
    scene_ids = []

    for f in track_files:
        match = re.match(r"(\d+)_tracks\.parquet", f.name)
        if match:
            scene_ids.append(match.group(1))
 
    return sorted(scene_ids)


def load_scene_data(
    results_dir: Path,
    scene_id: str,
    accident_frame: int,
    ego_involved: int,
    discretizer: ObservableDiscretizer,
    video_fps: float = 10.0
) -> pd.DataFrame:
    """
    Load and preprocess data for a single scene.

    Returns:
        DataFrame with discretized features, risk labels, and ego_involved label
    """
    tracks_path = results_dir / f"{scene_id}_tracks.parquet"
    env_path = results_dir / f"{scene_id}_env.parquet"

    if not tracks_path.exists():
        print(f"  Skipping {scene_id}: tracks file not found")
        return None

    # Load tracks
    df_tracks = pd.read_parquet(tracks_path)

    # Merge environment features if available
    if env_path.exists():
        df_env = pd.read_parquet(env_path)
        df_tracks = df_tracks.merge(df_env, on="frame", how="left")

    # Discretize features
    discrete_df = discretizer.transform(df_tracks)
    discrete_df = discretizer.encode_as_indices(discrete_df)
    discrete_df["frame"] = df_tracks["frame"].values
    discrete_df["scene_id"] = scene_id

    # Create TTA-based risk labels
    discrete_df["tta_frames"] = accident_frame - discrete_df["frame"]
    discrete_df["tta_seconds"] = discrete_df["tta_frames"] / video_fps

    def get_risk_label(tta):
        if tta >= 2.5:
            return 0  # SAFE
        elif tta >= 1.5:
            return 1  # ELEVATED
        else:
            return 2  # CRITICAL

    discrete_df["risk_label"] = discrete_df["tta_seconds"].apply(get_risk_label)

    # Add ego_involved label (same for all frames in scene)
    discrete_df["ego_involved"] = ego_involved

    return discrete_df


def prepare_frame_features(
    data: pd.DataFrame,
    feature_cols: List[str],
    label_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate features per frame per scene.

    Returns:
        Tuple of (frame_features DataFrame, available_cols list)
    """
    agg_dict = {col: "max" for col in feature_cols if col in data.columns}
    agg_dict[label_col] = "first"

    frame_features = data.groupby(["scene_id", "frame"]).agg(agg_dict).reset_index()
    available_cols = [col for col in feature_cols if col in frame_features.columns]

    return frame_features, available_cols


def train_risk_classifier(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    cv_folds: int = 5,
) -> Tuple[GradientBoostingClassifier, StandardScaler, List[str], Dict[str, Any]]:
    """
    Train GradientBoostingClassifier for risk level prediction.

    ML Standard approach:
    - Scaler is fit ONLY on training data (prevents data leakage)
    - Cross-validation performed on training data for model selection
    - Validation set used for early stopping and hyperparameter monitoring
    - Test set held out for final unbiased evaluation

    Returns:
        Tuple of (classifier, scaler, feature_cols, metrics_dict)
    """
    # Prepare training features
    train_features, available_cols = prepare_frame_features(
        train_data, feature_cols, "risk_label"
    )

    X_train = train_features[available_cols].fillna(0).values
    y_train = train_features["risk_label"].values

    # Fit scaler ONLY on training data (prevents data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Prepare validation features (transform only, no fit)
    val_features, _ = prepare_frame_features(val_data, feature_cols, "risk_label")
    X_val = val_features[available_cols].fillna(0).values
    y_val = val_features["risk_label"].values
    X_val_scaled = scaler.transform(X_val)

    # Prepare test features (transform only, no fit)
    test_features, _ = prepare_frame_features(test_data, feature_cols, "risk_label")
    X_test = test_features[available_cols].fillna(0).values
    y_test = test_features["risk_label"].values
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifier (no internal validation_fraction since we have explicit val set)
    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )

    # Cross-validation on training data for model selection
    print("  Running cross-validation on training data...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"  CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

    # Train final model on all training data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X_train_scaled, y_train)

    # Evaluate on all sets
    y_train_pred = classifier.predict(X_train_scaled)
    y_val_pred = classifier.predict(X_val_scaled)
    y_test_pred = classifier.predict(X_test_scaled)

    train_acc = (y_train_pred == y_train).mean()
    val_acc = (y_val_pred == y_val).mean()
    test_acc = (y_test_pred == y_test).mean()

    # Collect metrics
    metrics = {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "train_label_dist": {
            "safe": int((y_train == 0).sum()),
            "elevated": int((y_train == 1).sum()),
            "critical": int((y_train == 2).sum()),
        },
        "val_label_dist": {
            "safe": int((y_val == 0).sum()),
            "elevated": int((y_val == 1).sum()),
            "critical": int((y_val == 2).sum()),
        },
        "test_label_dist": {
            "safe": int((y_test == 0).sum()),
            "elevated": int((y_test == 1).sum()),
            "critical": int((y_test == 2).sum()),
        },
    }

    return classifier, scaler, available_cols, metrics


def train_ego_classifier(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    cv_folds: int = 5,
) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    Train GradientBoostingClassifier for ego-involved prediction.

    Uses the same scaler as risk classifier for consistency.
    Evaluates at SCENE level (not frame level) since ego_involved
    is a scene-level label.

    Returns:
        Tuple of (classifier, metrics_dict)
    """
    # Prepare training features (frame-level for training)
    train_features, available_cols = prepare_frame_features(
        train_data, feature_cols, "ego_involved"
    )

    X_train = train_features[available_cols].fillna(0).values
    y_train = train_features["ego_involved"].values
    X_train_scaled = scaler.transform(X_train)

    # Prepare validation features
    val_features, _ = prepare_frame_features(val_data, feature_cols, "ego_involved")
    X_val = val_features[available_cols].fillna(0).values
    y_val = val_features["ego_involved"].values
    X_val_scaled = scaler.transform(X_val)

    # Prepare test features
    test_features, _ = prepare_frame_features(test_data, feature_cols, "ego_involved")
    X_test = test_features[available_cols].fillna(0).values
    y_test = test_features["ego_involved"].values
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifier
    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )

    # Cross-validation on training data
    print("  Running cross-validation on training data...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"  CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

    # Train final model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X_train_scaled, y_train)

    # Evaluate at SCENE level (majority vote per scene) for proper evaluation
    def evaluate_scene_level(features_df, X_scaled, label=""):
        features_df = features_df.copy()
        features_df["pred"] = classifier.predict(X_scaled)
        scene_preds = features_df.groupby("scene_id").agg({
            "ego_involved": "first",
            "pred": lambda x: int(x.mean() >= 0.5)
        }).reset_index()
        y_true = scene_preds["ego_involved"].values
        y_pred = scene_preds["pred"].values
        acc = (y_true == y_pred).mean()
        return y_true, y_pred, acc, len(scene_preds)

    _, _, val_scene_acc, val_scenes = evaluate_scene_level(val_features, X_val_scaled, "val")
    y_scene_true, y_scene_pred, test_scene_acc, test_scenes = evaluate_scene_level(
        test_features, X_test_scaled, "test"
    )

    # Frame-level accuracy for comparison
    val_frame_acc = (y_val == classifier.predict(X_val_scaled)).mean()
    test_frame_acc = (y_test == classifier.predict(X_test_scaled)).mean()

    # Collect metrics
    metrics = {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "val_accuracy_scene_level": val_scene_acc,
        "val_accuracy_frame_level": val_frame_acc,
        "test_accuracy_scene_level": test_scene_acc,
        "test_accuracy_frame_level": test_frame_acc,
        "val_scenes": val_scenes,
        "test_scenes": test_scenes,
        "y_scene_true": y_scene_true,
        "y_scene_pred": y_scene_pred,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
    }

    return classifier, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train risk and ego-involved classifiers on multiple scenes"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing *_tracks.parquet and *_env.parquet files"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to Crash_Table.csv with accident metadata"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for trained classifier pickle file"
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=10.0,
        help="Video FPS for TTA calculation (default: 10.0)"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of boosting estimators (default: 100)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Max tree depth (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of scenes for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of scenes for testing (default: 0.15)"
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable visualization plots (default: show plots)"
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save visualization plots to disk"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MULTI-SCENE RISK & EGO-INVOLVED CLASSIFIER TRAINING")
    print("="*60)

    # Step 1: Get scene metadata (accident frames + ego_involved)
    print("\nLoading scene metadata...")
    scene_metadata = get_scene_metadata(args.meta_path)
    print(f"Found metadata for {len(scene_metadata)} scenes")

    ego_yes = sum(1 for m in scene_metadata.values() if m["ego_involved"] == 1)
    ego_no = len(scene_metadata) - ego_yes
    print(f"Ego-involved distribution: Yes={ego_yes}, No={ego_no}")

    # Step 2: Discover scenes in results directory
    print("\nDiscovering scenes...")
    scene_ids = discover_scenes(results_dir)
    print(f"Found {len(scene_ids)} scenes with track data")

    # Step 3: Split scenes into train/val/test (scene-level split prevents temporal leakage)
    print("\nSplitting scenes into train/validation/test sets...")
    print(f"  Target split: Train={1-args.val_size-args.test_size:.0%}, Val={args.val_size:.0%}, Test={args.test_size:.0%}")
    train_scene_ids, val_scene_ids, test_scene_ids = split_scenes_by_stratification(
        scene_ids, scene_metadata,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=42
    )
    print_split_statistics(train_scene_ids, val_scene_ids, test_scene_ids, scene_metadata)

    # Step 4: Initialize discretizer
    print("\nInitializing discretizer...")
    discretizer = ObservableDiscretizer.default()

    # Fit discretizer on first TRAINING scene to get feature definitions
    # (avoids data leakage from test scenes)
    first_scene = train_scene_ids[0]
    first_tracks = pd.read_parquet(results_dir / f"{first_scene}_tracks.parquet")
    first_env_path = results_dir / f"{first_scene}_env.parquet"

    if first_env_path.exists():
        first_env = pd.read_parquet(first_env_path)
        first_tracks = first_tracks.merge(first_env, on="frame", how="left")
    discretizer.fit(first_tracks)

    # Step 5: Load and process train and test scenes separately
    def load_scenes(scene_list: List[str], label: str) -> Tuple[List[pd.DataFrame], int]:
        """Load scenes from a list, returning data and count."""
        scene_data_list = []
        loaded = 0
        print(f"\nLoading {label} scenes...")
        for scene_id in scene_list:
            if scene_id not in scene_metadata:
                print(f"  Skipping {scene_id}: no metadata found")
                continue
            meta = scene_metadata[scene_id]
            scene_data = load_scene_data(
                results_dir, scene_id, meta["accident_frame"],
                meta["ego_involved"], discretizer, args.video_fps
            )
            if scene_data is not None:
                scene_data_list.append(scene_data)
                loaded += 1
                ego_str = "Yes" if meta["ego_involved"] else "No"
                print(f"  Loaded {scene_id}: {len(scene_data)} rows, "
                      f"accident@{meta['accident_frame']}, ego={ego_str}")
        return scene_data_list, loaded

    train_scene_data, train_count = load_scenes(train_scene_ids, "TRAIN")
    val_scene_data, val_count = load_scenes(val_scene_ids, "VALIDATION")
    test_scene_data, test_count = load_scenes(test_scene_ids, "TEST")

    if not train_scene_data:
        print("ERROR: No training scenes loaded. Check your data paths.")
        return
    if not val_scene_data:
        print("ERROR: No validation scenes loaded. Check your data paths.")
        return
    if not test_scene_data:
        print("ERROR: No test scenes loaded. Check your data paths.")
        return

    # Concatenate data
    train_data = pd.concat(train_scene_data, ignore_index=True)
    val_data = pd.concat(val_scene_data, ignore_index=True)
    test_data = pd.concat(test_scene_data, ignore_index=True)
    print(f"\nData summary:")
    print(f"  Train:      {len(train_data):>6} rows from {train_count} scenes")
    print(f"  Validation: {len(val_data):>6} rows from {val_count} scenes")
    print(f"  Test:       {len(test_data):>6} rows from {test_count} scenes")

    # Step 6: Get feature columns (discretized columns ending in _d)
    feature_cols = [col for col in train_data.columns if col.endswith("_d")]
    print(f"Feature columns: {len(feature_cols)}")

    # Step 7: Train risk classifier
    print("\n" + "-"*40)
    print("Training RISK classifier...")
    print("-"*40)
    risk_classifier, scaler, used_feature_cols, risk_metrics = train_risk_classifier(
        train_data,
        val_data,
        test_data,
        feature_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        cv_folds=args.cv_folds,
    )

    # Print risk classifier results
    print(f"\n  Results:")
    print(f"    Train accuracy:      {risk_metrics['train_accuracy']:.1%}")
    print(f"    Validation accuracy: {risk_metrics['val_accuracy']:.1%}")
    print(f"    TEST accuracy:       {risk_metrics['test_accuracy']:.1%}")
    print(f"    Samples - Train: {risk_metrics['train_samples']}, Val: {risk_metrics['val_samples']}, Test: {risk_metrics['test_samples']}")
    print(f"    Train dist: Safe={risk_metrics['train_label_dist']['safe']}, "
          f"Elevated={risk_metrics['train_label_dist']['elevated']}, "
          f"Critical={risk_metrics['train_label_dist']['critical']}")
    print(f"    Val dist:   Safe={risk_metrics['val_label_dist']['safe']}, "
          f"Elevated={risk_metrics['val_label_dist']['elevated']}, "
          f"Critical={risk_metrics['val_label_dist']['critical']}")
    print(f"    Test dist:  Safe={risk_metrics['test_label_dist']['safe']}, "
          f"Elevated={risk_metrics['test_label_dist']['elevated']}, "
          f"Critical={risk_metrics['test_label_dist']['critical']}")

    # Print classification report for test set
    print("\n  Test Set Classification Report:")
    print(classification_report(
        risk_metrics['y_test'], risk_metrics['y_test_pred'],
        target_names=['Safe', 'Elevated', 'Critical']
    ))

    # Step 8: Train ego-involved classifier
    print("\n" + "-"*40)
    print("Training EGO-INVOLVED classifier...")
    print("-"*40)
    ego_classifier, ego_metrics = train_ego_classifier(
        train_data,
        val_data,
        test_data,
        used_feature_cols,
        scaler,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        cv_folds=args.cv_folds,
    )

    # Print ego classifier results
    print(f"\n  Results:")
    print(f"    Validation accuracy (scene-level): {ego_metrics['val_accuracy_scene_level']:.1%}")
    print(f"    TEST accuracy (scene-level):       {ego_metrics['test_accuracy_scene_level']:.1%}")
    print(f"    Validation accuracy (frame-level): {ego_metrics['val_accuracy_frame_level']:.1%}")
    print(f"    TEST accuracy (frame-level):       {ego_metrics['test_accuracy_frame_level']:.1%}")
    print(f"    Scenes - Val: {ego_metrics['val_scenes']}, Test: {ego_metrics['test_scenes']}")

    # Print classification report for test set (scene-level)
    print("\n  Test Set Classification Report (Scene-Level):")
    print(classification_report(
        ego_metrics['y_scene_true'], ego_metrics['y_scene_pred'],
        target_names=['Not-Involved', 'Involved']
    ))

    # Step 9: Visualize results
    show_plots = not args.no_plots
    save_path = output_path if args.save_plots else None

    if show_plots or args.save_plots:
        print("\n" + "-"*40)
        print("Generating validation visualizations...")
        print("-"*40)

        # Main validation results plot
        visualize_results(
            risk_classifier=risk_classifier,
            ego_classifier=ego_classifier,
            risk_metrics=risk_metrics,
            ego_metrics=ego_metrics,
            feature_cols=used_feature_cols,
            output_path=save_path,
            show_plots=show_plots,
        )

        # Class distribution plot
        visualize_class_distributions(
            risk_metrics=risk_metrics,
            train_scenes=train_count,
            test_scenes=test_count,
            output_path=save_path,
            show_plots=show_plots,
        )

    # Step 10: Save classifiers and scaler
    print(f"\nSaving models to {output_path}...")
    model_data = {
        "risk_classifier": risk_classifier,
        "ego_classifier": ego_classifier,
        "scaler": scaler,
        "feature_cols": used_feature_cols,
        "video_fps": args.video_fps,
        "n_scenes_trained": train_count,
        "n_scenes_tested": test_count,
        "n_frames_trained": risk_metrics['train_samples'],
        "n_frames_tested": risk_metrics['test_samples'],
        "risk_metrics": {
            "cv_mean": float(risk_metrics['cv_mean']),
            "cv_std": float(risk_metrics['cv_std']),
            "train_accuracy": float(risk_metrics['train_accuracy']),
            "test_accuracy": float(risk_metrics['test_accuracy']),
        },
        "ego_metrics": {
            "cv_mean": float(ego_metrics['cv_mean']),
            "cv_std": float(ego_metrics['cv_std']),
            "test_accuracy_scene_level": float(ego_metrics['test_accuracy_scene_level']),
            "test_accuracy_frame_level": float(ego_metrics['test_accuracy_frame_level']),
        },
        # Keep backward compatibility
        "classifier": risk_classifier,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {output_path}")
    print(f"  - Train scenes: {train_count}, Test scenes: {test_count}")
    print(f"  - Train frames: {risk_metrics['train_samples']}, Test frames: {risk_metrics['test_samples']}")
    print(f"  - Features: {len(used_feature_cols)}")
    print(f"  - Risk classifier:")
    print(f"      CV accuracy: {risk_metrics['cv_mean']:.1%} (+/- {risk_metrics['cv_std']*2:.1%})")
    print(f"      TEST accuracy: {risk_metrics['test_accuracy']:.1%}")
    print(f"  - Ego-involved classifier:")
    print(f"      CV accuracy: {ego_metrics['cv_mean']:.1%} (+/- {ego_metrics['cv_std']*2:.1%})")
    print(f"      TEST accuracy (scene-level): {ego_metrics['test_accuracy_scene_level']:.1%}")


if __name__ == "__main__":
    main()
