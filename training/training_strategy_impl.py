from abc import ABC, abstractmethod
from typing import Any
import os
class TrainingStrategy(ABC):
    @abstractmethod
    def validate_environment(self):
        pass

    def run(self, model, config, observers):
        self.validate_environment()
        return self.train(model, config, observers)
    
    def train(self, model, config, observers) -> Any:
            for obs in observers:
                obs.update("start", {"model": model.model})

            results =  model.train(
                data=config["data_yaml"],
                epochs=config["epochs"],
                imgsz=config["imgsz"],
                batch=config["batch"],
                project=config["project"],
                name=config["experiment_name"]
            )

            for obs in observers:
                obs.update("end", {})
            return results

class SingleProcessTrainingStrategy(TrainingStrategy):   
    def validate_environment(self):
        return 
    
class DistributedDDPTrainingStrategy(TrainingStrategy):
    def validate_environment(self):
        required_vars = ["RANK", "WORLD_SIZE","LOCAL_RANK"]
        missing = [v for v in required_vars if v not in os.environ]
        if missing:
            raise RuntimeError(
                "DistributedDDPTrainingStrategy selected but"
                "torchrun environment variables are missing. "
                "Did you forget to launch with torchrun?"
            )
        return 
    
