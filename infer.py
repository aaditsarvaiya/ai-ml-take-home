"""Run the trained YOLO model on a folder of images.

Saves annotated images + one .txt per image:
    class_id x_center y_center width height confidence   (all normalized)

Usage:
    python infer.py --input /path/to/images --output /path/to/out
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="folder of input images")
    parser.add_argument("--output", required=True, help="folder for annotated images + txt files")
    parser.add_argument("--weights", default="runs/detect/runs/baseline_yolov8n/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.weights)
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in in_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for img_path in sorted(images):
        results = model.predict(str(img_path), conf=args.conf, verbose=False)[0]

        # annotated image (plot() returns BGR already)
        cv2.imwrite(str(out_dir / img_path.name), results.plot())

        # txt: class_id xc yc w h conf, normalized
        lines = []
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls = int(b.cls[0])
                xc, yc, w, h = b.xywhn[0].tolist()
                conf = float(b.conf[0])
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}")
        (out_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    print(f"done: {len(images)} images -> {out_dir}")


if __name__ == "__main__":
    main()