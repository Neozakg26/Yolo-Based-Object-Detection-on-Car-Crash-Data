import pandas as pd
import numpy as np

class EnvironmentBuilder:

    @staticmethod
    def build(df_tracks: pd.DataFrame) -> pd.DataFrame:

        env = df_tracks.groupby("frame").agg(
            # distance_proxy: larger = farther, so min gives closest object
            min_distance_t=("distance_proxy", "min"),
            mean_rel_speed_t=("risk_speed", "mean"),
            min_ttc_t=("ttc_proxy", "min"),
            # proximity: larger = closer, so > threshold gives close objects
            num_objects_close_t=("proximity", lambda x: (x > 0.1).sum())
        ).reset_index()

        # Replace infinite TTC with large cap (ML-friendly)
        env["min_ttc_t"] = env["min_ttc_t"].replace(np.inf, 999)

        return env
