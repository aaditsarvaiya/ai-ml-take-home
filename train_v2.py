from ultralytics import YOLO

# Custom-augmentation v2 — the ACTUAL config used for the documented v2 run.
# (An earlier committed version of this file carried 80-epoch v3 params that
# were never run; this restores the 40-epoch config from METHODOLOGY.md.)
model = YOLO("yolov8n.pt")

model.train(
    data="Sampled-YD-Object-Detection-2/data.yaml",
    epochs=40,
    imgsz=512,
    project="runs",
    name="custom_aug",
    degrees=20,
    hsv_h=0.025,
    hsv_s=0.5,
    hsv_v=0.65,
    translate=0.15,
    erasing=0.25,
)
