import argparse
import pandas as pd
import re

from explainability.hierarchical_dbn import AccidentRiskAssessor
from explainability.metadata import MetaData
from training.config_loader import ConfigLoader

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to global DBN model parquet (global_risk_model.parquet)")
parser.add_argument("--track_path", type=str, required=True)
parser.add_argument("--env_path", type=str, required=True)

parser.add_argument("--inference_method", type=str, default="supervised",
                    choices=["supervised", "belief_propagation", "variable_elimination"])
args = parser.parse_args()

# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")
BASE_PATH = config.paths["cluster_base"]
META_PATH = f"{BASE_PATH}/Crash_Table.csv"

# ---------- LOAD METADATA ----------
scene_id = re.search(r"\d+", args.track_path).group().zfill(6)
meta = MetaData(META_PATH, scene_id)

df_tracks = pd.read_parquet(args.track_path)
env_df = pd.read_parquet(args.env_path)

# ---------- LOAD GLOBAL MODEL ----------
print("\n" + "="*60)
print("GLOBAL MODEL INFERENCE")
print("="*60)

risk_assessor = AccidentRiskAssessor.load(args.model_path)
risk_assessor.inference_method = args.inference_method  # To override the inference
print("Loaded global model.")

# ---------- RUN INFERENCE ----------
print("Running risk inference...")
risk_trajectory = risk_assessor.get_risk_trajectory(df_tracks, env_df)

# ---------- SAVE OUTPUT ----------
risk_path = f"{BASE_PATH}/results/{scene_id}_risk_trajectory.parquet"
risk_trajectory.to_parquet(risk_path, index=False, engine="pyarrow")
print(f"Risk trajectory saved: {risk_path}")

# ---------- PRINT SUMMARY ----------
print("\nRisk Assessment Summary:")
print(f"Total frames assessed: {len(risk_trajectory)}")
for risk_level in ["Safe", "Elevated", "Critical"]:
    count = (risk_trajectory["MAP_Risk"] == risk_level).sum()
    pct = count / len(risk_trajectory) * 100
    print(f"  {risk_level}: {count} frames ({pct:.1f}%)")

critical_frames = risk_trajectory[risk_trajectory["P_Critical"] > 0.5]
if len(critical_frames) > 0:
    print(f"\nCritical risk detected in {len(critical_frames)} frames")
    print(f"First critical frame: {critical_frames['frame'].min()}")
    print(f"Peak critical probability: {critical_frames['P_Critical'].max():.2%}")

if "MAP_EgoInvolved" in risk_trajectory.columns:
    print("\nEgo-Involved Prediction:")
    mean_p_ego = risk_trajectory["P_EgoInvolved_Yes"].mean()
    overall_ego = "Yes" if mean_p_ego > 0.5 else "No"
    print(f"  Overall prediction: Ego-Involved={overall_ego} (mean P={mean_p_ego:.1%})")

    if "egoinvolve" in meta.metadata:
        gt_ego = meta.metadata["egoinvolve"]
        match = "CORRECT" if (gt_ego == overall_ego) else "INCORRECT"
        print(f"  Ground truth: {gt_ego} [{match}]")

# Accident lead-time check (clip-frame indexing caution: depends on your MetaData)
if "accident_start_frame" in meta.metadata and pd.notna(meta.metadata["accident_start_frame"]):
    accident_frame = int(meta.metadata["accident_start_frame"])
    print(f"\nGround truth accident start (metadata): frame {accident_frame}")

    pre_accident = risk_trajectory[risk_trajectory["frame"] < accident_frame]
    if len(pre_accident) > 0:
        elevated_before = pre_accident[(pre_accident["P_Elevated"] + pre_accident["P_Critical"]) > 0.5]
        if len(elevated_before) > 0:
            first_warning = elevated_before["frame"].min()
            warning_lead = accident_frame - first_warning
            print(f"  First elevated risk warning: frame {first_warning} ({warning_lead} frames before accident)")

print("\nPipeline complete.")
