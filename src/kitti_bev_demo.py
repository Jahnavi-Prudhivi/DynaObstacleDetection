"""
YOLO + ByteTrack + BEV projection on a KITTI drive.

Every ByteTrack ID is projected to the ground and drawn in the top-down map
with the SAME colour and ID as in the camera view, as a real-size footprint
with its motion trail.

Keys while running:  w = toggle warped road background   q = quit

Output:
  outputs/kitti_bev.mp4      camera view (left) + metric BEV map (right)
  outputs/tracks_bev.csv     per-frame, per-track ground positions (for motion estimation)
"""

import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bev import GroundPlaneBEV, load_p2


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "kitti" / "images"
CALIB_FILE = PROJECT_ROOT / "data" / "kitti" / "2011_09_26 3" / "calib_cam_to_cam.txt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "kitti_bev.mp4"
OUTPUT_CSV = OUTPUT_DIR / "tracks_bev.csv"


# --------------------------------------------------
# Config
# --------------------------------------------------

FPS = 10
TRAIL_LENGTH = 30
CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
CONF_THRESHOLD = 0.3
BOTTOM_MARGIN_PX = 3     # boxes touching the image bottom have a clipped foot point
SHOW_ROAD_WARP = False   # warped road texture behind the map (toggle with 'w')

# BEV map extent: 25 m each side, 60 m ahead, 10 px per metre
BEV_X_RANGE = (-25, 25)
BEV_Z_RANGE = (0, 60)
BEV_PX_PER_M = 10


# --------------------------------------------------
# Setup
# --------------------------------------------------

model = YOLO("yolo11n.pt")
bev = GroundPlaneBEV(load_p2(CALIB_FILE), x_range=BEV_X_RANGE,
                     z_range=BEV_Z_RANGE, px_per_m=BEV_PX_PER_M)

image_paths = sorted(IMAGE_DIR.glob("*.png"))
if not image_paths:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

first_frame = cv2.imread(str(image_paths[0]))
height, width = first_frame.shape[:2]
print(f"Found {len(image_paths)} frames ({width}x{height}). Horizon row ~ {bev.horizon_v:.1f}px")

# BEV panel scaled to the camera image height, placed on the right
panel_w = int(round(bev.bev_w * height / bev.bev_h))

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width + panel_w, height)
)

csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2",
     "foot_u", "foot_v", "X_m", "Z_m", "valid"]
)


def color_for_id(track_id: int) -> tuple:
    """Deterministic BGR colour per track ID (shared by camera view and BEV)."""
    rng = np.random.default_rng(track_id * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))   # image-space trail
ground_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))  # ground-space trail (X, Z)
unique_ids = set()


# --------------------------------------------------
# Main loop
# --------------------------------------------------

for frame_number, image_path in enumerate(image_paths):
    frame = cv2.imread(str(image_path))
    if frame is None:
        continue

    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        classes=CLASSES, conf=CONF_THRESHOLD, verbose=False,
    )

    annotated = frame.copy()
    canvas = bev.make_canvas(frame if SHOW_ROAD_WARP else None)

    # horizon line: road markings should converge here if the calibration is right
    hv = int(round(bev.horizon_v))
    cv2.line(annotated, (0, hv), (width, hv), (0, 255, 255), 1)

    n_tracked = n_on_map = n_invalid = 0
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        # Project every box's bottom-centre (road contact point) to the ground at once
        foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
        ground, valid = bev.image_to_ground(foot)
        valid &= xyxy[:, 3] < height - BOTTOM_MARGIN_PX

        for i, (track_id, cls_id, conf) in enumerate(zip(track_ids, class_ids, confs)):
            n_tracked += 1
            x1, y1, x2, y2 = xyxy[i].astype(int)
            X, Z = ground[i]
            ok = bool(valid[i])
            cls_name = model.names[cls_id]
            color = color_for_id(track_id)
            unique_ids.add(track_id)

            csv_writer.writerow(
                [frame_number, track_id, cls_name, f"{conf:.3f}",
                 x1, y1, x2, y2, f"{foot[i, 0]:.1f}", f"{foot[i, 1]:.1f}",
                 f"{X:.3f}" if ok else "", f"{Z:.3f}" if ok else "", int(ok)]
            )

            # ---------- camera view ----------
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            dist = f" {Z:.1f}m" if ok else ""
            label = f"ID {track_id} {cls_name}{dist}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            track_history[track_id].append((int((x1 + x2) / 2), y2))
            pts = list(track_history[track_id])
            for k in range(1, len(pts)):
                cv2.line(annotated, pts[k - 1], pts[k], color, max(1, int(3 * k / len(pts))))

            # ---------- BEV view (same ID, same colour) ----------
            if not ok:
                n_invalid += 1
                continue
            ground_history[track_id].append((X, Z))
            if bev.draw_object(canvas, track_id, cls_name, list(ground_history[track_id]), color):
                n_on_map += 1

    # ---------- HUDs ----------
    cv2.putText(annotated, f"Frame: {frame_number}   Unique IDs: {len(unique_ids)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"tracked {n_tracked} | on map {n_on_map} | no ground pt {n_invalid}",
                (6, bev.bev_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    panel = cv2.resize(canvas, (panel_w, height), interpolation=cv2.INTER_AREA)
    combined = np.hstack([annotated, panel])
    video_writer.write(combined)

    cv2.imshow("KITTI - YOLO + ByteTrack + BEV", combined)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key == ord("q"):
        break
    if key == ord("w"):
        SHOW_ROAD_WARP = not SHOW_ROAD_WARP


video_writer.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Total unique tracked objects: {len(unique_ids)}")
print(f"Video: {OUTPUT_VIDEO}\nTracks CSV: {OUTPUT_CSV}")