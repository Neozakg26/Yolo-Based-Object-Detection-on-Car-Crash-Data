import yaml
from dataclasses import dataclass 
from core.mylogger import MyLogger

# Instantiate logger globally
logger = MyLogger("converter_config.log")

@dataclass
class Config:
    images_dir: str
    annotations_dir: str
    output_labels_dir: str
    copier: str


class ConverterConfigLoader:
    @staticmethod
    def load(path: str) -> Config:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            logger.error_log(f"Invalid  converter config file loaded: {path}")
            raise ValueError(f"Converter Config file is empty or invalid: {path}")    

        return Config(
            images_dir=data["images_dir"],
            annotations_dir=data["annotations_dir"],
            output_labels_dir=data["labels_dir"],
            copier=data["copier"]
        )
