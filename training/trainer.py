from typing import Any
from training.config_loader import Config
from training.distributed_context import DistributedContext

class Trainer:
    def __init__(self, model, config, trainer_service, distributed_context):
        self.model = model
        self.config = config
        self.trainer_service = trainer_service
        self.dist = distributed_context

    # validate environment
    # load config, model and return results only using the  trainer_service
    # only master process (rank == 0) will write logs. Others have empty observers
    def run(self, observers) -> Any: 
        self.trainer_service.validate_env() 
        active_observers = observers if self.dist.is_master else []
        return self.trainer_service.train(self.model,self.config,active_observers) 
    

    
