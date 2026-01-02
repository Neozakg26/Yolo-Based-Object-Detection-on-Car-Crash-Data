from training.api import (
    DistributedContext,
    ConfigLoader,
    ModelFactory,
    Trainer,
    LoggerObserver,
    MyLogger,
    TrainerService,
    StrategySelector,
    TrainConfigBuilder,
)

from validation.api import (
    Validator,
    ValidationStrategySelector,
    ValidationConfigBuilder,
    ValidationService,
    MetricCalculator
)

import os
import torch.distributed as dist
import argparse

def maybe_init_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 0:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True)
    args = parser.parse_args()

    #Load Config 
    config = ConfigLoader.load("config.yaml")

    if args.job == "val":
        global_logger = MyLogger(log_file="validation.log") #if dist.is_master else None
        service = ValidationService(
            selected_strategy=ValidationStrategySelector.select(config.compute["validation_strategy"]),
            config=ValidationConfigBuilder.build(config=config),
            metric_calculator=MetricCalculator    ###FIX the metric Calc
        )

        #Load Validator 
        validator = Validator(
            model= ModelFactory.load(config.model["best"]),
            validation_service= service
        )

        # only add Logger Observsers to Master 
        observers =[]
        #if dist.is_master: 
        observers.append(global_logger)
            
        validator.run(observers=observers)

    
    elif args.job == "train": 

        print("Executing Training")
        maybe_init_distributed()

        dist = DistributedContext()

        global_logger = MyLogger(log_file="training.log") if dist.is_master else None
        
        service  = TrainerService(
            selected_strategy= StrategySelector().select(config.compute["training_strategy"]),                      
            train_config_builder= TrainConfigBuilder())
        
        #Load trainer
        trainer = Trainer(
            model = ModelFactory().load(config.model["name"]),
            config = config,
            trainer_service = service,
            distributed_context= dist
        )

        # only add Logger Observsers to Master 
        observers =[]
        if dist.is_master: 
            observers.append(LoggerObserver(logger=global_logger))

        results  = trainer.run(observers)

        if dist.is_master: 
            best_path  = f"{results.save_dir}/weights/best.pt"
            print(f"Neo best model: {best_path}")
    else:
        print("No valid Job Specified!!")


