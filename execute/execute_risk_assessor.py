from explainability.feature_extractor import FeatureExtractor
from explainability.hierarchical_dbn import AccidentRiskAssessor
from explainability.metadata import MetaData
from training.config_loader import ConfigLoader
import argparse
import pickle
import pandas as pd
import re 

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--features_path", type=str, required=True)
parser.add_argument("--track_path", type=str, required=True)
parser.add_argument("--env_path", type=str, required=True)

parser.add_argument("--skip_dbn", action="store_true", help="Skip hierarchical DBN risk assessment")
parser.add_argument("--inference_method", type=str, default="supervised",
                    choices=["supervised", "belief_propagation", "variable_elimination"],
                    help="Inference method: supervised (recommended), belief_propagation, or variable_elimination")
parser.add_argument("--pretrained_classifier", type=str, default=None,
                    help="Path to pre-trained classifier pickle file. If provided, skips per-scene training.")
args = parser.parse_args()

# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")

BASE_PATH = config.paths["cluster_base"]
META_PATH = f"{BASE_PATH}/Crash_Table.csv"
# ---------- LOAD METADATA ----------
scene_id = re.search(r'\d+', args.track_path).group()
meta = MetaData(META_PATH, scene_id)

# ---------- LOAD FEATURES GRAPH  ----------
with open(f"{args.features_path}","rb") as f:
    graph = pickle.load(f) 

df_tracks = pd.read_parquet(args.track_path)
env_df = pd.read_parquet(args.env_path)

# ---------- HIERARCHICAL DBN RISK ASSESSMENT ----------
print("\n" + "="*60)
print("HIERARCHICAL DBN RISK ASSESSMENT")
print("="*60)

# Initialize risk assessor
risk_assessor = AccidentRiskAssessor(
    inference_method=args.inference_method,
    prior_strength=10.0,
    pretrained_classifier_path=args.pretrained_classifier
)

# Fit the model with tracks, environment, metadata, and causal structure
print("Fitting hierarchical DBN...")
risk_assessor.fit(
    tracks_df=df_tracks,
    env_df=env_df,
    metadata=meta.metadata,
    pcmci_graph=graph
)

# Assess risk for the entire video
print("Running risk inference...")
risk_trajectory = risk_assessor.get_risk_trajectory(df_tracks, env_df)

# Save risk trajectory
risk_path = f"{BASE_PATH}/results/{scene_id}_risk_trajectory.parquet"
risk_trajectory.to_parquet(risk_path, index=False, engine="pyarrow")
print(f"Risk trajectory saved: {risk_path}")

# Save trained model
model_path = f"{BASE_PATH}/results/{scene_id}_risk_model.parquet"
risk_assessor.save(model_path)
print(f"Risk model saved: {model_path}")

# Print risk summary
print("\nRisk Assessment Summary:")
print(f"Total frames assessed: {len(risk_trajectory)}")
print(f"Risk distribution:")
for risk_level in ["Safe", "Elevated", "Critical"]:
    count = (risk_trajectory["MAP_Risk"] == risk_level).sum()
    pct = count / len(risk_trajectory) * 100
    print(f"    {risk_level}: {count} frames ({pct:.1f}%)")

# Find critical moments
critical_frames = risk_trajectory[risk_trajectory["P_Critical"] > 0.5]
if len(critical_frames) > 0:
    print(f"Critical risk detected in {len(critical_frames)} frames")
    print(f"First critical frame: {critical_frames['frame'].min()}")
    print(f"Peak critical probability: {critical_frames['P_Critical'].max():.2%}")

# Print ego-involved prediction summary if available
if "MAP_EgoInvolved" in risk_trajectory.columns:
    print("\nEgo-Involved Prediction:")
    for ego_val in ["Yes", "No"]:
        count = (risk_trajectory["MAP_EgoInvolved"] == ego_val).sum()
        pct = count / len(risk_trajectory) * 100
        print(f"    {ego_val}: {count} frames ({pct:.1f}%)")

    # Overall ego-involved prediction (majority vote or mean probability)
    mean_p_ego = risk_trajectory["P_EgoInvolved_Yes"].mean()
    overall_ego = "Yes" if mean_p_ego > 0.5 else "No"
    print(f"  Overall prediction: Ego-Involved={overall_ego} (mean P={mean_p_ego:.1%})")

    # Compare with ground truth if available
    if "egoinvolve" in meta.metadata:
        gt_ego = meta.metadata["egoinvolve"]
        match = "CORRECT" if (gt_ego == overall_ego) else "INCORRECT"
        print(f"  Ground truth: {gt_ego} [{match}]")

# Compare with metadata accident frame if available
if "accident_start_frame" in meta.metadata and pd.notna(meta.metadata["accident_start_frame"]):
    accident_frame = int(meta.metadata["accident_start_frame"])
    print(f"\nGround truth accident start: frame {accident_frame}")

    # Check if model detected elevated/critical risk before accident
    pre_accident = risk_trajectory[risk_trajectory["frame"] < accident_frame]
    if len(pre_accident) > 0:
        elevated_before = pre_accident[pre_accident["P_Elevated"] + pre_accident["P_Critical"] > 0.5]
        if len(elevated_before) > 0:
            first_warning = elevated_before["frame"].min()
            warning_lead = accident_frame - first_warning
            print(f"  First elevated risk warning: frame {first_warning} ({warning_lead} frames before accident)")

print("\nPipeline complete.")