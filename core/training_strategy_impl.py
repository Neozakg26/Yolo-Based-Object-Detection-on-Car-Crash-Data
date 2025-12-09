from abc import ABC, abstractmethod
from core.mylogger import MyLogger
# import torch
# import pynvml
from typing import Any

class StrategySelector(ABC):
    @abstractmethod
    def train(self, model, config, observers):
        pass

# class SingleGPUTrainingStrategy and MultiGPUTrainingStrategy 
# inherit Abstract classs TrainingStrategy

class SingleGPUTrainingStrategy(StrategySelector):   
    def train(self, model, config, observers) -> Any:
        for obs in observers:
            obs.update("start", {"model": model.model})

        results =  model.train(
            data=config["data_yaml"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            project=config["project"],
            name=config["experiment_name"]
        )

        for obs in observers:
            obs.update("end", {})
        return results

# class MultiGPUTrainingStrategy(StrategySelector): #Select top N idle GPU's 
#     def __init__(self, max_gpus=2, min_free_memory_gb=4):
#         self.max_gpus = max_gpus
#         self.min_free_memory_gb = min_free_memory_gb
    
#     def select_idle_gpus(self):
#         pynvml.nvmlInit()
#         gpu_count = pynvml.nvmlDeviceGetCount()
#         gpu_info = []

#         for i in range(gpu_count):
#             handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#             util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#             mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#             mem_free_gb = mem_info.free / 1024**3
#             num_procs = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle) or [])
#             if mem_free_gb < self.min_free_memory_gb:
#                 continue
#             score = util * 0.6 + (1 - mem_free_gb / (mem_info.total / 1024**3)) * 100 * 0.3 + num_procs * 10
#             gpu_info.append((score, i))

#         pynvml.nvmlShutdown()
#         if not gpu_info:
#             logger.warning("No suitable idle GPUs found. Using CPU.")
#             return ["cpu"]

#         gpu_info.sort()
#         selected = [gpu_id for (_, gpu_id) in gpu_info[:self.max_gpus]]
#         logger.info(f"🎯 Selected idle GPUs: {selected}")
#         return ",".join(str(g) for g in selected)

#     def train(self, model, config, observers):
#         for obs in observers:
#             obs.update("start", {"model": model.model})
#         device =self.select_idle_gpus()
#         model.train(
#             data=config["data_yaml"],
#             epochs=config["epochs"],
#             imgsz=config["imgsz"],
#             batch=config["batch"],
#             project=config["project"],
#             name=config["experiment_name"],
#             device=device   
#         )

#         for obs in observers:
#             obs.update("end", {})
