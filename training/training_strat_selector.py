import torch
from typing import Any 

from .training_strategy_impl import (
    SingleGPUTrainingStrategy
 #   ,MultiGPUTrainingStrategy
)

# #Available If I want to initialise a specific training context and train via it.
# class TrainingContext:
#     def __init__(self, strategy):
#         self.strategy = strategy

#     def set_strategy(self, strategy):
#         self.strategy = strategy

#     def train(self, model, config, observers) -> Any:
#         return self.strategy.train(model, config, observers)


class StrategySelector:
    @staticmethod
    def select(strategy_name: str):
        if strategy_name == "single":
            return SingleGPUTrainingStrategy()
        elif strategy_name == "multi":
            return SingleGPUTrainingStrategy()
        else:
            # AUTO SELECT
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("Auto-Select: Using Multi-GPU Strategy")
                return SingleGPUTrainingStrategy() ## TODO: Change to Parallel Trainer after adding it 
            else:
                print("Auto-Select: Using Single-GPU Strategy")
                return SingleGPUTrainingStrategy()
