class ValidationConfigBuilder:
    @staticmethod
    def build(config):  
        return {
            "data_yaml": config.dataset["yaml"],
            "imgsz": config.val["imgsz"],
            "batch": config.val["batch"],
            "split": config.val["split"],
            "conf": config.val["conf"],
            "iou": config.val["iou"],
            "save_model": config.val["save_model"],
        }
