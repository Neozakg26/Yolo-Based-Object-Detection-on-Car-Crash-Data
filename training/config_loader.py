import yaml
from dataclasses import dataclass 
from training.mylogger import MyLogger

# Instantiate logger globally
logger = MyLogger("training.log")

@dataclass
class Config:
    model: dict
    dataset: dict
    train: dict
    compute: dict

class ConfigLoader:
    @staticmethod
    def load(path: str) -> Config:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            logger.log(f"Loaded config from {path}: {data}" )
        if data is None:
            logger.error_log(f"Invalid config file loaded: {path}")
            raise ValueError(f"Config file is empty or invalid: {path}")    

        return Config(
            model=data["model"],
            dataset=data["dataset"],
            train=data["train"],
            compute=data["compute"]
        )
