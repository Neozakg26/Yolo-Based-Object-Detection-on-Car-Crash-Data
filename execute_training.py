from core.config_loader import ConfigLoader
from core.model_factory import ModelFactory
from core.trainer import Trainer
from core.observer import LoggerObserver #, GPUObserver
from core.mylogger import MyLogger
from core.trainer_service import TrainerService
from core.training_strat_selector import StrategySelector
from core.trainer_conf_builder import TrainConfigBuilder
# import threading

if __name__ == "__main__":

    global_logger = MyLogger(log_file="training.log")
    config = ConfigLoader.load("config.yaml")
    service  = TrainerService(
        strategy_selector= StrategySelector(),                      
        train_config_builder= TrainConfigBuilder())
    
    #Load trainer
    trainer = Trainer(
        model_factory = ModelFactory(),
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

        if config.train["save_model"]:
            print(f"Saving model post training")
            
        print(f"Type Object returned {type(results)}")
        print(f"results: {results}")
        print(f" metrics {results.names}")
        print(f"save dir {results.nt_per_class}")


    finally:
        # gpu_observer.stop()
        # gpu_thread.join()
        global_logger.log("GPU observer stopped.")

