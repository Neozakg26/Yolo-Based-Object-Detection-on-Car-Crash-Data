class TrainConfigBuilder:
    @staticmethod
    def build(config):
        return {
            "data_yaml": config.dataset["yaml"],
            "epochs": config.train["epochs"],
            "imgsz": config.train["imgsz"],
            "batch": config.train["batch"],
            "project": config.train["project"],
            "experiment_name": config.train["experiment_name"],
            "save_model": config.train["save_model"]
        } 
