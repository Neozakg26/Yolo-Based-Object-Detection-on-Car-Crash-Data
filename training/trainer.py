from typing import Any
from training.config_loader import Config

class Trainer:
    def __init__(self, model_factory, config, trainer_service):
        self.model_factory = model_factory
        self.config = config
        self.trainer_service = trainer_service

    # load config, model and return results from trainer_service
    def run(self, observers) -> Any: 
        model = self.model_factory.load(self.config.model["name"])
        return self.trainer_service.train(model,self.config,observers) 
    

    
