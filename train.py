from ultralytics import YOLO

# yolov8n = nano: smallest YOLOv8, fastest to train, good baseline
model = YOLO("yolov8n.pt")

model.train(
    data="Sampled-YD-Object-Detection-2/data.yaml",
    epochs=40,
    imgsz=512,
    project="runs",
    name="baseline_yolov8n",
)
