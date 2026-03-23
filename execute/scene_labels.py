from pathlib import Path
import pandas as pd
from training.config_loader import ConfigLoader

# ---------- CONFIG ----------
config = ConfigLoader.load("config.yaml")

acc_root = Path(config.paths["cluster_base"])
norm_root = Path(config.paths["cluster_normal_images"])

scene_ids = set()
rows=[]
for d in sorted([p for p in acc_root.iterdir() if p.is_file()]):
    scene= d.stem.rsplit("_",1)[0]
    if scene not in scene_ids:
        scene_ids.add(scene)
        rows.append({"scene_id": d.stem.rsplit("_",1)[0], "label": 1})

for d in sorted([p for p in norm_root.iterdir() if p.is_file()]):
    scene= d.stem.rsplit("_",1)[0]
    if scene not in scene_ids:
        scene_ids.add(scene)
        rows.append({"scene_id": d.stem.rsplit("_",1)[0], "label": 0})

df = pd.DataFrame(rows)
df.to_csv("scene_index.csv", index=False)
# df[["scene_id","label"]].to_csv("scene_labels.csv", index=False)
# print("Wrote scene_index.csv and scene_labels.csv with", len(df), "scenes")

print(df)