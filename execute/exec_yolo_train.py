from training.config_loader import ConfigLoader
from training.model_factory import ModelFactory
from training.trainer import Trainer
from training.observer import LoggerObserver #, GPUObserver
from training.mylogger import MyLogger
from training.trainer_service import TrainerService
from training.training_strat_selector import StrategySelector
from training.trainer_conf_builder import TrainConfigBuilder

from validation.validator import Validator
from validation.validation_service import ValidationService
from validation.validation_strat_selector import ValidationStrategySelector
from validation.validation_conf_builder import ValidationConfigBuilder
from validation.metric_calculator import MetricCalculator
# import threading

if __name__ == "__main__":

    global_logger = MyLogger(log_file="training.log")
    config = ConfigLoader.load("config.yaml")
    service  = TrainerService(
        selected_strategy= StrategySelector().select(config.compute["training_strategy"]),                      
        train_config_builder= TrainConfigBuilder())
    
    #Load trainer
    trainer = Trainer(
        model = ModelFactory().load(config.model["name"]),
        config = config,
        trainer_service = service
    )

#   Inject logger instance into observer 
    logger_observer = LoggerObserver(logger=global_logger)
#   gpu_observer = GPUObserver(interval_sec=30)
    observers = [logger_observer]

    # GPU Observer thread
    # gpu_thread = threading.Thread(target=gpu_observer.start)
    # gpu_thread.start()

    try:
        results  = trainer.run(observers)

        best_path  = f"{results.save_dir}/weights/best.pt"
        print(f"Neo best model: {best_path}")

        if config.train["save_model"]:
            print(f"Saving model post training")
        
        print(f"Type Object returned {type(results)}")

        print(f"metrics {results.names}")
        print(f"save dir {results.nt_per_class}")

        #Initialise validator and load saved model to validate 
        validate_service = ValidationService(
            selected_strategy= ValidationStrategySelector().select(config.compute["validation_strategy"]),
            config= ValidationConfigBuilder().build(config),
            metric_calculator= MetricCalculator()
        )

        validator  = Validator(
            model= ModelFactory().load(best_path),
            validation_service= validate_service
        )
        validate_results = validator.run(observers=observers)
    finally:
        # gpu_observer.stop()
        # gpu_thread.join()
        global_logger.log("GPU observer stopped.")

