from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "kitti" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "kitti_yolo_demo.mp4"


# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------------------------
# Get all KITTI frames in chronological order
# --------------------------------------------------

image_paths = sorted(IMAGE_DIR.glob("*.png"))

if not image_paths:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

print(f"Found {len(image_paths)} KITTI frames.")


# --------------------------------------------------
# Read first frame to determine video dimensions
# --------------------------------------------------

first_frame = cv2.imread(str(image_paths[0]))

if first_frame is None:
    raise FileNotFoundError(f"Could not load {image_paths[0]}")

height, width = first_frame.shape[:2]


# --------------------------------------------------
# Create output video
# --------------------------------------------------

fps = 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# Process every KITTI frame
# --------------------------------------------------

for frame_number, image_path in enumerate(image_paths):

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Skipping {image_path}")
        continue

    # Run YOLO
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Write frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Add frame to output video
    video_writer.write(annotated_frame)

    # Display it
    cv2.imshow("KITTI - YOLO Detection", annotated_frame)

    # Wait based on desired playback FPS.
    # Press q to quit early.
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Clean up
# --------------------------------------------------

video_writer.release()
cv2.destroyAllWindows()

print(f"Video saved to: {OUTPUT_VIDEO}")