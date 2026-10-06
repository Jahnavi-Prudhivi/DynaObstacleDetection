from pathlib import Path

import cv2
from ultralytics import YOLO


# Get the root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the first KITTI image
image_path = PROJECT_ROOT / "data" / "kitti" / "images" / "0000000000.png"

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Read image
image = cv2.imread(str(image_path))

# Make sure image loaded correctly
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Run YOLO object detection
results = model(image)

# Draw bounding boxes and labels
annotated_image = results[0].plot()

# Display result
cv2.imshow("KITTI - YOLO Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()