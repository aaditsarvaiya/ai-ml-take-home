import cv2
from pathlib import Path

video = "/Users/aadit/Downloads/bottle_mallet.MOV"
out = Path("/tmp/video_frames")
out.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video)
i = saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i % 10 == 0:
        cv2.imwrite(str(out / f"frame_{saved:04d}.jpg"), frame)
        saved += 1
    i += 1
cap.release()
print(f"saved {saved} frames -> {out}")