from typing import Any
from training.config_loader import Config

class Trainer:
    def __init__(self, model, config, trainer_service):
        self.model = model
        self.config = config
        self.trainer_service = trainer_service

    # load config, model and return results from trainer_service
    def run(self, observers) -> Any: 
        return self.trainer_service.train(self.model,self.config,observers) 
    

    
