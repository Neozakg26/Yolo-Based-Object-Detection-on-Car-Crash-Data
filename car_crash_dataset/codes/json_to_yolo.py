import json
import os
from PIL import Image
from pathlib import Path
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Process, Queue, cpu_count
from .converter_config_loader import ConverterConfigLoader
import time

# --- Configuration ---

config = ConverterConfigLoader.load("car_crash_dataset/codes/converter_config.yaml")
IMAGES_DIR = Path(config.images_dir)
ANNOTATIONS_DIR = Path(config.annotations_dir)
OUTPUT_LABELS_DIR = Path(config.output_labels_dir)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_LABELS_DIR / "conversion_progress.txt"

#Mapping category to class index
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "person": 3,
    "bike": 4,
    "traffic sign":5,
}



# --- Logging setup ---
def setup_listener(queue: Queue, log_file="json_converter_logs.log"):
    handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    listener = QueueListener(queue, handler)
    listener.start()
    return listener

def get_logger(name: str, queue: Queue):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.INFO)
    return logger

# --- Conversion function ---
def convert_annotation(annotation_file: Path, queue: Queue, completed_files , retries=3):
    logger = get_logger(f"worker-{annotation_file.stem}", queue)

    # Skip already processed files
    if annotation_file.name in completed_files:
        logger.info(f"Skipping {annotation_file.name} (already converted)")
        return
    
    # queue_handler = QueueHandler(queue)
    # logger.addHandler(queue_handler)
    # logger.setLevel(logging.INFO)

    for attempt in range(1, retries + 1):
        try:
            with open(annotation_file,"r") as f:
                data = json.load(f)

            image_name = data.get("name")
            frames = data.get("frames",[])

            if not image_name:
                logger.warning(f"No image name found in {annotation_file.name}, skipping.")
                return

            image_path = IMAGES_DIR / f"{image_name}.jpg"
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping.")
                return
            
            img = Image.open(image_path)
            img_w, img_h = img.size
            results = []

            for frame in frames:
                for obj in frame.get("objects",[]):
                    category = obj.get("category","UNKNOWN")
                    if "box2d" not in obj or obj["category"] not in CLASS_MAP:
                        logger.info(f"Category Not Found: {obj['category']}")
                        continue

                    x1, y1 = obj["box2d"]["x1"], obj["box2d"]["y1"]
                    x2, y2 = obj["box2d"]["x2"], obj["box2d"]["y2"]

                    # Avoid division by zero
                    if img_w == 0 or img_h == 0:
                        logger.error(f"Invalid image size for {image_name}, skipping.")
                        continue

                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    class_id = CLASS_MAP[category]
                    results.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Write YOLO label file
            out_file = OUTPUT_LABELS_DIR / f"{image_name}.txt"
            with open(out_file, "w") as f:
                f.write("\n".join(results))

            # Update checkpoint
            completed_files.add(annotation_file.name)

            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                f.write(annotation_file.name + "\n")

            logger.info(f"Converted {annotation_file.name} → {out_file.name}")

        except Exception as e:
            logger.error(f"Error processing {annotation_file.name} (Attempt {attempt}): {e}")
            time.sleep(1) 
    logger.error(f"Failed to convert {annotation_file.name} after {retries} attempts.")


# --- Worker function using a task queue ---
def worker(queue: Queue, task_queue: Queue, completed_files):
    logger = get_logger(f"worker-{os.getpid()}", queue)
    while True:
        try:
            annotation_file = task_queue.get_nowait()
        except Exception:
            break  # Queue empty
        convert_annotation(annotation_file, queue, completed_files=completed_files)


if __name__ == "__main__":

        # --- Checkpoint file to track progress ---

    if CHECKPOINT_FILE.exists():
        print("Found completed_files")
        with open(CHECKPOINT_FILE, "r") as f:
            completed_files = {line.strip() for line in f if line.strip()}
    else:
        print("completed_files doesn't exist")
        with open(CHECKPOINT_FILE, "w") as f:
            pass
        completed_files = set()


    logger = logging.getLogger("json_to_yolo.logger")

    try:
        json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
        print(f"Starting conversion of {len(json_files)} files")
        
        # Create task queue
        task_queue = Queue()
        for f in json_files:
            if f.name not in completed_files:
                task_queue.put(f)
        
        if task_queue.empty():
            print("All files already converted.")
            exit(0)

        print(f"{task_queue.qsize()} files in Queue")

        # Logging queue and listener
        log_queue = Queue()
        listener = setup_listener(log_queue)

        # Determine workers
        num_workers = min(cpu_count(), task_queue.qsize())
        print(f"Using {num_workers} worker processes.")

        # Start worker processes
        processes = []
        for i in range(num_workers):
            p = Process(target=worker , args=(log_queue, task_queue,completed_files))
            p.start()
            processes.append(p)

        # Wait for all workers
        for p in processes:
            print(f"completing process :{p.name}")
            p.join()
        print("All files processed.")
        logging.getLogger("main").info("Parallel annotation conversion finished.")

    except Exception as e:
        logging.getLogger("main").error(f"Fatal error in conversion pipeline: {e}")

    finally:
        listener.stop()
        print("Logging listener stopped.")
