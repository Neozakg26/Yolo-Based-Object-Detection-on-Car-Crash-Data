from typing import Any 

from .training_strategy_impl import (
    SingleProcessTrainingStrategy,
      DistributedDDPTrainingStrategy
)

class StrategySelector:
    @staticmethod
    def select(strategy_name: str):
        if strategy_name == "single":   
            return SingleProcessTrainingStrategy()
        elif strategy_name == "multi":
            return DistributedDDPTrainingStrategy()
        else:
            raise ValueError(f"Unknown training strategy: {strategy_name}")