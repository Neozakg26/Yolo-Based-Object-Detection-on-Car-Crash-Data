import pandas as pd
class MetaData:
 
    DTYPES = {
    "vidname": "string",
    "weather": "string",
#    "frame": "int32",
    "timing": "string",
    "egoinvolve": "string",
#    "accident_start_frame": "int32",
    }

    def __init__(self,path, scene_id):
        df_meta = pd.read_csv(
            path,
            dtype=self.DTYPES
        )
        
        # print(f"df_meta: {df_meta}")
        # print(f"df_meta: {df_meta}")
        frame_cols = [c for c in df_meta.columns if c.startswith("frame_")]
        #print(f"frame_cols: {frame_cols}")

        df_long = df_meta.melt(
            id_vars=["vidname", "egoinvolve","timing","weather"],
            value_vars=frame_cols,
            var_name="frame"
            ,value_name="accident_flag"
        )

        # print(f"df_long: {df_long}")

        
        df_long["frame"] = (
            df_long["frame"]
            .str.replace("frame_", "", regex=False)
            .astype("int32")
        )
       
        df_long = df_long.astype({
            "frame": "int32",
            "accident_flag": "int8"
        })

        # print(f"df_long2: {df_long}")
        df_long = (
            df_long
            .set_index(["vidname", "frame"])
            .sort_index()
        )

        #print(f"df_long3: {df_long}")
        df_scene_meta = (
            df_long
            .reset_index()
            .groupby("vidname")[["egoinvolve","timing","weather"]]
            .first()
        )

 

        accident_start = (
            df_long
            .query("accident_flag == 1")
            .groupby(level="vidname")
            .apply(lambda g: g.index.get_level_values("frame")
            .min())
            .rename("accident_start_frame")
        )

        # print(f"accident_start: {accident_start}")

        df_scene_meta = df_scene_meta.join(accident_start, how="left")
        self.metadata = df_scene_meta.loc[scene_id].to_dict()

    

        
 

        