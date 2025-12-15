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


from validation.validator import Validator
from validation.validation_service import ValidationService
from validation.validation_strat_selector import ValidationStrategySelector
from validation.validation_conf_builder import ValidationConfigBuilder
from validation.metric_calculator import MetricCalculator
# import threading

if __name__ == "__main__":

    dist = DistributedContext()

    global_logger = MyLogger(log_file="training.log") if dist.is_master else None
    config = ConfigLoader.load("config.yaml")
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
        # validate_service = ValidationService(
        #     selected_strategy= ValidationStrategySelector().select(config.compute["validation_strategy"]),
        #     config= ValidationConfigBuilder().build(config),
        #     metric_calculator= MetricCalculator()
        # )

        # validator  = Validator(
        #     model= ModelFactory().load(best_path),
        #     validation_service= validate_service
        # )
        # validate_results = validator.run(observers=observers)


