"""
YOLO + ByteTrack + BEV projection on a KITTI raw drive.

BEV improvements over the first version:
  - skip boxes cut off at the bottom of the image (no road-contact point -> bogus distance)
  - hide objects far to the side (LATERAL_LIMIT)
  - correct for the car pitching (braking/bumps) using OXTS, if available
  - light smoothing of each object's BEV position to remove frame-to-frame jitter
  - feed ByteTrack low-confidence boxes and use a larger input size -> fewer dropouts

Keys: w = toggle warped road, q = quit
"""

import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bev import GroundPlaneBEV, load_p2, load_pitch

# ---------------- paths ----------------
ROOT = Path(__file__).resolve().parent.parent
KITTI = ROOT / "data" / "kitti"
IMAGE_DIR = KITTI / "images"                                  # <drive>_sync/image_02/data
CALIB_FILE = KITTI / "2011_09_26 3" / "calib_cam_to_cam.txt"
OXTS_DIR = KITTI / "oxts"                                     # <drive>_sync/oxts (optional)
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------- config ----------------
CLASSES = [0, 1, 2, 3, 5, 7]   # person, bicycle, car, motorcycle, bus, truck
DET_CONF = 0.1                 # keep low: ByteTrack uses 0.1-0.3 boxes to bridge dropouts
DRAW_CONF = 0.3                # boxes below this are tracked but not drawn
IMGSZ = 1280                   # KITTI is 1242 px wide; default 640 shrinks distant cars
LATERAL_LIMIT = 6.0            # m left/right of the camera to show (your lane + one each side)
SMOOTHING = 0.4                # 0..1, weight of the new position (1.0 = no smoothing)
JUMP_RESET = 5.0               # m: bigger jumps reset the smoothing instead of averaging
TRAIL_LENGTH = 30
FPS = 20

# ---------------- setup ----------------
model = YOLO("yolo11s.pt")
bev = GroundPlaneBEV(load_p2(CALIB_FILE))
images = sorted(IMAGE_DIR.glob("*.png"))
if not images:
    raise FileNotFoundError(f"No PNG images in {IMAGE_DIR}")
pitch = load_pitch(OXTS_DIR, len(images))
H, W = cv2.imread(str(images[0])).shape[:2]
panel_w = int(bev.bev_w * H / bev.bev_h)

video = cv2.VideoWriter(str(OUT / "kitti_bev.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W + panel_w, H))
csv_file = open(OUT / "tracks_bev.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "raw_X", "raw_Z", "X", "Z"])

smoothed = {}                                         # track_id -> smoothed [X, Z]
trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
show_road = False


def color_for_id(i):
    return tuple(int(c) for c in np.random.default_rng(int(i) * 9973).integers(60, 255, 3))


# ---------------- main loop ----------------
for k, path in enumerate(images):
    frame = cv2.imread(str(path))
    res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=CLASSES,
                      conf=DET_CONF, imgsz=IMGSZ, verbose=False)[0]
    annotated, canvas = frame.copy(), bev.make_canvas(frame if show_road else None)
    hv = int(bev.horizon(pitch[k]))
    cv2.line(annotated, (0, hv), (W, hv), (0, 255, 255), 1)

    if res.boxes.id is not None:
        for box, tid, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist(),
                                          res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            if y2 > H - 3:              # cut off at the bottom: foot point unknown
                continue
            xz, ok = bev.image_to_ground([[(x1 + x2) / 2, y2]], pitch[k])
            X, Z = xz[0]
            if not ok[0] or abs(X) > LATERAL_LIMIT:
                continue

            # smoothing: blend with the previous position unless it jumped (new object / bad box)
            prev = smoothed.get(tid)
            new = np.array([X, Z])
            if prev is not None and np.linalg.norm(new - prev) < JUMP_RESET:
                new = SMOOTHING * new + (1 - SMOOTHING) * prev
            smoothed[tid] = new
            trails[tid].append(new)

            name = model.names[cls_id]
            writer.writerow([k, tid, name, f"{conf:.3f}", *np.round(box, 1),
                             f"{X:.3f}", f"{Z:.3f}", f"{new[0]:.3f}", f"{new[1]:.3f}"])
            if conf < DRAW_CONF:
                continue

            color = color_for_id(tid)
            label = f"ID {tid} {name} {new[1]:.1f}m"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            bev.draw_object(canvas, tid, name, list(trails[tid]), color)

    cv2.putText(annotated, f"Frame {k}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    combined = np.hstack([annotated, cv2.resize(canvas, (panel_w, H), interpolation=cv2.INTER_AREA)])
    video.write(combined)
    cv2.imshow("KITTI BEV", combined)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key == ord("q"):
        break
    if key == ord("w"):
        show_road = not show_road

video.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Saved {OUT / 'kitti_bev.mp4'} and {OUT / 'tracks_bev.csv'}")