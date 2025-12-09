import json
import os
from PIL import Image
from pathlib import Path
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from multiprocessing import Process, Queue
from .converter_config_loader import ConverterConfigLoader


config = ConverterConfigLoader.load("car_crash_dataset/codes/converter_config.yaml")
# Define Global paths and Create target Dir
IMAGES_DIR = Path(config.images_dir)
ANNOTATIONS_DIR = Path(config.annotations_dir)
OUTPUT_LABELS_DIR = Path(config.images_dir)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

#Mapping category to class index
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "person": 3,
    "bike": 4,
    "traffic sign":5,
}

def convert_annotation(annotation_file: Path, queue: Queue):
    logger = logging.getLogger(f"worker-{annotation_file.stem}")
    queue_handler = QueueHandler(queue)
    logger.addHandler(queue_handler)
    logger.setLevel(logging.INFO)

    try:
        with open(annotation_file,"r") as f:
            data = json.load(f)

        image_name = data["name"]
        frames = data["frames"]

        image_path = os.path.join(IMAGES_DIR, image_name + ".jpg")
        img = Image.open(image_path)
        img_w, img_h = img.size

        results = []
        for frame in frames:
            for obj in frame["objects"]:
                if "box2d" not in obj or obj["category"] not in CLASS_MAP:
                    logger.info(f"Category Not Found: {obj}")
                    continue
                x1, y1 = obj["box2d"]["x1"], obj["box2d"]["y1"]
                x2, y2 = obj["box2d"]["x2"], obj["box2d"]["y2"]

                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                class_id = CLASS_MAP[obj["category"]]
                results.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        out_file = OUTPUT_LABELS_DIR / f"{image_name}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join(results))

        logger.info(f"Converted {annotation_file.name} → {out_file.name}")

    except Exception as e:
        logger.error(f"Error in {annotation_file.name}: {e}")


def parallel_convert(json_files, queue: Queue):
    for annotation_file in json_files:
        convert_annotation(annotation_file, queue)

def worker_task(queue, worker_id):
    logger = logging.getLogger(f"worker-{worker_id}")
    logger.addHandler(QueueHandler(queue))  # Send logs to central queue
    logger.setLevel(logging.INFO)

    for i in range(3):
        logger.info(f"Processing step {i} in worker {worker_id}")
        # time.sleep(0.5)

def setup_listener(queue):
    handler = logging.FileHandler("parallel_logs.log",encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    listener = QueueListener(queue, handler)
    listener.start()
    return listener



if __name__ == "__main__":

    logger = logging.getLogger("json_to_yolo.logger")

    try:
        json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
        print(f"Starting parallel conversion of {len(json_files)} files")
        
        num_workers = min(len(json_files),multiprocessing.cpu_count())  

        total_files = len(json_files)
        max_workers = multiprocessing.cpu_count()
        num_workers = min(total_files, max_workers)
        files_per_worker = max(1, total_files // num_workers)

        queue = Queue()
        listener = setup_listener(queue)

        # Start worker processes
        processes = []

        for i in range(num_workers):
            start = i * files_per_worker
            end = start + files_per_worker if i < num_workers - 1 else total_files
            worker_files = json_files[start:end]

            # The worker now includes logger queue + file allocation
            p = Process(target=parallel_convert , args=(worker_files, queue))
            p.start()
            processes.append(p)

        # Wait for all workers
        for p in processes:
            p.join()

        logger.info(f"Process Execute Order: {processes}")
        logger.info("Parallel annotation conversion finished!")

    except Exception as e:
        logger.error(f"Fatal error in conversion pipeline: {e}")

    finally:
        listener.stop()
        logger.info("Logging listener stopped")
        print("All processes finished logging")