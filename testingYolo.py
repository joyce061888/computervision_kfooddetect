from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Paths to two images
image_paths = ["./data/bibimbap.png", "./data/tokbokki.png"]

# Run detection on images (confidence score of 30% - only keep detections where the confidence is ≥ 30%)
results = model.predict(image_paths, conf=0.6)

# Loop through each image and its result
for idx, (path, result) in enumerate(zip(image_paths, results)):
    image = cv2.imread(path)

    # Draw bounding boxes
    for box in result.boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # Class ID
        cls = int(box.cls[0].item())
        # Confidence score
        conf = float(box.conf[0].item())
        # Class name (e.g. 'pizza', 'bowl')
        label = model.names[cls]

        # Draw rectangle + label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show each result
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detections for {path}")
    plt.axis("off")

plt.show()
