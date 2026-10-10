from ultralytics import YOLO

model = YOLO("runs/detect/runs/baseline_yolov8n/weights/best.pt")
model.predict("Sampled-YD-Object-Detection-2/valid/images", save=True)
