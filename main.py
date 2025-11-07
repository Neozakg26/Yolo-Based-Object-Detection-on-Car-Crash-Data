
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# # Train the model on your custom dataset
# print("Hey there I am trainig Yolo")
# model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)

results = model("C:\\Users\\neokg\\Coding_Projects\\CCD_videos\\0A_poTqyb1c.mp4", save=True, show =True)