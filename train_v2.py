from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Sampled-YD-Object-Detection-2/data.yaml",
    epochs=80,
    imgsz=512,
    project="runs",
    name="custom_aug",
    degrees=30.0,
    hsv_h=0.028,
    hsv_s=0.55,
    hsv_v=0.55,
    translate=0.13,
    erasing=0.19,
)
