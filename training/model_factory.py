from ultralytics import YOLO

class ModelFactory:
    @staticmethod
    def load(model_name: str):
        #FactoryClass to create Yolo Model from Specified model i.e. ModelFactory.create("yolo11n.pt")
        return YOLO(model_name)
