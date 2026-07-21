from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Ctrl-Find AI Backend")

# Load your trained Glasses Detection model
model = YOLO("ultralytics/runs/detect/train-3/weights/best.pt")


@app.get("/")
def home():
    return {
        "status": "Working",
        "project": "Ctrl-Find",
        "message": "Backend Running Successfully"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()

    # Convert bytes to OpenCV image
    image = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "status": "error",
            "message": "Invalid Image"
        }

    # Run detection
    results = model(frame)

    objects = []

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        objects.append({
            "name": model.names[cls],
            "confidence": round(conf, 2)
        })

    return {
        "status": "success",
        "total_objects": len(objects),
        "objects": objects
    }