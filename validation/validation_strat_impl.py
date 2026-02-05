class SimpleValidationStrategy:
    def validate(self, model, config, observers):
        results = model.val(
            data=config["data_yaml"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            split=config["split"],
            conf= config["conf"],
            iou= config["iou"]
        )

        # predictions = results.pred
        # targets = results.targets

        for obs in observers:
            obs.log(f"Finished Validation: {results}")

        return results
