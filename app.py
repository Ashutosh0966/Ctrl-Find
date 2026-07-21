from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "Working",
        "message": "YOLO Backend Started Successfully"
    }
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model only once
model = YOLO("yolov8n.pt")


@app.get("/")
def home():
    return {"status": "Working"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model(frame)

    detections = []

    for box in results[0].boxes:

        cls = int(box.cls)

        conf = float(box.conf)

        detections.append({
            "object": model.names[cls],
            "confidence": round(conf, 2)
        })

    return {
        "detections": detections
    }
