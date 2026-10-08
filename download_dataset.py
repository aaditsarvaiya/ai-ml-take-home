import truststore
truststore.inject_into_ssl()

api_key = None
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("ROBOFLOW_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip("'\"")
            break
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY not found in .env")

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(2)
dataset = version.download("yolov8")
print("downloaded to:", dataset.location)
