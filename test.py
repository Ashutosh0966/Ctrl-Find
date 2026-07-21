import cv2

for i in range(5):
    print(f"Testing Camera {i}...")

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera {i} Working")
        else:
            print(f"Camera {i} Opened but No Frame")

        cap.release()
    else:
        print(f"Camera {i} Not Found")

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Phone camera stream
url = "http://192.168.1.11:8080/video"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    # YOLO detection
    results = model(frame)

    # Draw boxes
    annotated = results[0].plot()

    cv2.imshow("YOLO Phone Camera", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

url = "http://192.168.1.11:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot connect to phone camera!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received")
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Phone Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
