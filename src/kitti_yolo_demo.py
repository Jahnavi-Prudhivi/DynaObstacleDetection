from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "kitti" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "kitti_yolo_bytetrack.mp4"


# --------------------------------------------------
# Config
# --------------------------------------------------

FPS = 10
TRAIL_LENGTH = 30  # number of past frames kept per track

# COCO class ids relevant to driving scenes:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
CLASSES = [0, 1, 2, 3, 5, 7]

CONF_THRESHOLD = 0.3


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

first_frame = cv2.imread(str(image_paths[0]))

if first_frame is None:
    raise FileNotFoundError(f"Could not load {image_paths[0]}")

height, width = first_frame.shape[:2]

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (width, height),
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def color_for_id(track_id: int) -> tuple:
    """Deterministic, visually distinct BGR color per track ID."""
    rng = np.random.default_rng(track_id * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


# Track history: id -> deque of (cx, cy) bottom-center points
# (bottom-center of the box approximates where the object touches the road)
track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
unique_ids = set()


# --------------------------------------------------
# Process every frame with ByteTrack
# --------------------------------------------------

for frame_number, image_path in enumerate(image_paths):

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Skipping {image_path}")
        continue

    # persist=True keeps tracker state between calls (required for frame-by-frame input)
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=CLASSES,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    annotated_frame = frame.copy()
    boxes = results[0].boxes

    # boxes.id is None when no tracks are active in this frame
    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        for (x1, y1, x2, y2), track_id, cls_id, conf in zip(
            xyxy, track_ids, class_ids, confs
        ):
            unique_ids.add(track_id)
            color = color_for_id(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label: ID + class name + confidence
            label = f"ID {track_id} {model.names[cls_id]} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Update and draw motion trail
            track_history[track_id].append((int((x1 + x2) / 2), y2))

            points = list(track_history[track_id])
            for i in range(1, len(points)):
                # Older segments are thinner so the trail fades visually
                thickness = max(1, int(3 * i / len(points)))
                cv2.line(annotated_frame, points[i - 1], points[i], color, thickness)

    # HUD
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}   Unique IDs so far: {len(unique_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    video_writer.write(annotated_frame)

    cv2.imshow("KITTI - YOLO + ByteTrack", annotated_frame)

    if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Clean up
# --------------------------------------------------

video_writer.release()
cv2.destroyAllWindows()

print(f"Total unique tracked objects: {len(unique_ids)}")
print(f"Video saved to: {OUTPUT_VIDEO}")