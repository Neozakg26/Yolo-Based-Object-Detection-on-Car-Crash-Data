from .mylogger import MyLogger
# import pynvml
import time

class TrainingObserver:
    def update(self, event: str, data: dict):
        pass

class LoggerObserver(TrainingObserver):
    def __init__(self, logger: MyLogger):
        self.logger = logger

    """Logs events using global logger"""
    def update(self, event: str, data: dict):
        messages = {
            "start": f" Training started for model: {data.get('model', 'unknown')}",
            "epoch_end": f" Epoch {data.get('epoch', '?')} completed.",
            "end": "Training completed successfully!",
        }
        msg = messages.get(event, f"Event {event} received.")
        self.logger.log(msg)

# class GPUObserver(TrainingObserver):
#     """Logs GPU utilization periodically during training"""
#     def __init__(self, interval_sec=30):
#         self.interval_sec = interval_sec
#         self.running = False

#     def start(self):
#         pynvml.nvmlInit()
#         self.running = True
#         while self.running:
#             gpu_count = pynvml.nvmlDeviceGetCount()
#             usage_list = []
#             for i in range(gpu_count):
#                 handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#                 util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#                 mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#                 mem_free_gb = mem_info.free / 1024**3
#                 mem_total_gb = mem_info.total / 1024**3
#                 usage_list.append(f"GPU {i}: util {util}%, free {mem_free_gb:.1f}GB/{mem_total_gb:.1f}GB")
#             my_logger.info(" | ".join(usage_list))
#             time.sleep(self.interval_sec)

#     def stop(self):
#         self.running = False
#         pynvml.nvmlShutdown()
