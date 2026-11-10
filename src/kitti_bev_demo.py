"""
YOLO + ByteTrack + BEV projection on a KITTI raw drive.

BEV improvements over the first version:
  - skip boxes cut off at the bottom of the image (no road-contact point -> bogus distance)
  - hide objects far to the side in the BEV (LATERAL_LIMIT)
  - correct for the car pitching (braking/bumps) using OXTS, if available
  - light smoothing of each object's BEV position to remove frame-to-frame jitter
  - feed ByteTrack low-confidence boxes and use a larger input size -> fewer dropouts

Fixes in this version:
  - stable class per track (confidence-weighted vote): YOLO flips a truck between
    truck / car / bus from frame to frame, and the BEV footprint size is looked up
    from the class, so the footprint jumped between 4.5, 8 and 11 m long
  - earlier detection of distant objects:
      * ByteTrack may now start tracks from weaker boxes (default needs conf >= 0.25)
      * a track stays drawn once confirmed, even when one frame's confidence dips
      * the image box is drawn even when the object is not put in the BEV
        (to the side, cut off at the bottom, or above the horizon)
      * a bigger model (yolo11m) finds small distant cars much more reliably
  - START_FRAME to skip the boring start, and no artificial delay between frames
  - fewer ID switches when a far object comes close:
      * fuse_score off: ByteTrack multiplied IoU by the detection score, so a far box
        with conf 0.2-0.3 could almost never be matched to its own track
      * slightly looser IoU gate and longer memory for lost tracks
      * BEV re-linking: a brand-new ID that appears where a recently lost track
        should be (in metres, using its BEV velocity) inherits the old ID

Keys: w = toggle warped road, space = pause, q = quit
"""

import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import ultralytics
import yaml
from ultralytics import YOLO

from bev import GroundPlaneBEV, camera_pitch_from_calib, load_p2, load_pitch

# ---------------- paths ----------------
ROOT = Path(__file__).resolve().parent.parent
KITTI = ROOT / "data" / "kitti"
IMAGE_DIR = KITTI / "image_00" / "data"                                  # <drive>_sync/image_02/data
CALIB_FILE = KITTI / "2011_09_26 3" / "calib_cam_to_cam.txt"
OXTS_DIR = KITTI / "oxts"                                     # <drive>_sync/oxts (optional)
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------- config ----------------
START_FRAME = 100              # skip the first frames (nothing happens before ~200)
DELAY_MS = 1                   # wait between frames in the window; 1 = as fast as possible
MODEL = "yolo11m.pt"           # 's' misses small distant cars; use "yolo11s.pt" if too slow
CLASSES = [0, 1, 2, 3, 5, 7]   # person, bicycle, car, motorcycle, bus, truck
DET_CONF = 0.1                 # keep low: ByteTrack uses 0.1-0.3 boxes to bridge dropouts
IMGSZ = 1280                   # KITTI is 1242 px wide; try 1600 to upscale tiny far objects
# ByteTrack settings (overrides of Ultralytics' bytetrack.yaml)
TRACKER = {
    "track_high_thresh": 0.2,  # default 0.25: 1st association pass
    "new_track_thresh": 0.2,   # default 0.25: min conf to START a track (far cars often 0.15-0.25)
    "track_low_thresh": 0.1,
    "fuse_score": False,       # default True: match cost = 1 - IoU*conf -> weak far boxes never match
    "match_thresh": 0.85,      # default 0.8: accept IoU > 0.15 (small far boxes shift a lot per frame)
    "track_buffer": 45,        # default 30: frames a lost track is kept for re-matching (4.5 s)
}
# BEV re-linking of new IDs to recently lost tracks
STITCH_MAX_GAP = 20            # frames a lost track can be re-linked (2 s at 10 Hz)
STITCH_WINDOW = 5              # a new ID can be re-linked during its first N frames
CLASS_GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle",
               "bicycle": "bike", "motorcycle": "bike", "person": "person"}
DRAW_CONF = 0.25               # a track is drawn once its best confidence reached this ...
MIN_HITS = 2                   # ... and it was seen in at least this many frames
LATERAL_LIMIT = 6.0            # m left/right of the camera to show in the BEV
SMOOTHING = 0.4                # 0..1, weight of the new position (1.0 = no smoothing)
USE_CALIB_PITCH = True         # correct the camera's fixed tilt (from the 3 calib files)
JUMP_RESET = 5.0               # m: bigger jumps reset the smoothing instead of averaging
CLASS_DECAY = 0.95             # class vote memory per frame (lower = adapts faster)
TRAIL_LENGTH = 30
FPS = 20                       # output video frame rate (KITTI is 10 Hz, so 20 = 2x speed)


def make_tracker_yaml(overrides, path):
    """Copy Ultralytics' default bytetrack.yaml with our overrides (keeps any version-specific keys)."""
    cfg = yaml.safe_load(open(Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"))
    cfg.update(overrides)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


# ---------------- setup ----------------
model = YOLO(MODEL)
tracker_cfg = make_tracker_yaml(TRACKER, OUT / "bytetrack_kitti.yaml")
static_pitch = camera_pitch_from_calib(CALIB_FILE.parent) if USE_CALIB_PITCH else 0.0
print(f"Camera tilt used: {np.degrees(static_pitch):+.2f} deg (+ = down)")
bev = GroundPlaneBEV(load_p2(CALIB_FILE), static_pitch=static_pitch)
images = sorted(IMAGE_DIR.glob("*.png"))
if not images:
    raise FileNotFoundError(f"No PNG images in {IMAGE_DIR}")
if START_FRAME >= len(images):
    raise ValueError(f"START_FRAME={START_FRAME} but the drive only has {len(images)} frames")
pitch = load_pitch(OXTS_DIR, len(images))             # indexed by the real frame number
H, W = cv2.imread(str(images[0])).shape[:2]
panel_w = int(bev.bev_w * H / bev.bev_h)

video = cv2.VideoWriter(str(OUT / "kitti_bev.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W + panel_w, H))
csv_file = open(OUT / "tracks_bev.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "track_id", "raw_track_id", "class", "raw_class", "conf", "x1", "y1", "x2", "y2",
                 "raw_X", "raw_Z", "X", "Z"])

smoothed = {}                                         # track_id -> smoothed [X, Z]
trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
class_votes = defaultdict(lambda: defaultdict(float))  # track_id -> {cls_id: decayed conf sum}
hits = defaultdict(int)                               # track_id -> frames seen
best_conf = defaultdict(float)                        # track_id -> highest conf seen
raw_hits = defaultdict(int)                           # ByteTrack id -> frames seen
alias = {}                                            # ByteTrack id -> displayed id
last_seen = {}                                        # displayed id -> last frame seen
velocity = {}                                         # displayed id -> BEV velocity [m/frame]
show_road = False


def color_for_id(i):
    return tuple(int(c) for c in np.random.default_rng(int(i) * 9973).integers(60, 255, 3))


def stable_class(tid, cls_id, conf):
    """Confidence-weighted, slowly decaying class vote, so one odd frame can't flip the class."""
    votes = class_votes[tid]
    for c in votes:
        votes[c] *= CLASS_DECAY
    votes[cls_id] += conf
    return max(votes, key=votes.get)


def stitch(raw_xz, name, k, busy):
    """Displayed id of a recently lost track whose predicted BEV position matches raw_xz, else None.

    The gate grows with distance because flat-ground depth error grows roughly with Z^2.
    """
    best, best_d = None, 1.0
    for did, seen in last_seen.items():
        gap = k - seen
        if did in busy or not 0 < gap <= STITCH_MAX_GAP or did not in smoothed:
            continue
        if CLASS_GROUP.get(model.names[max(class_votes[did], key=class_votes[did].get)]) != CLASS_GROUP.get(name):
            continue
        pred = smoothed[did] + velocity.get(did, np.zeros(2)) * gap
        z = max(pred[1], raw_xz[1], 1.0)
        gate = np.array([1.5 + 0.04 * z, 2.0 + 0.2 * z])        # [lateral, depth] in metres
        d = np.linalg.norm((raw_xz - pred) / gate)
        if d < best_d:
            best, best_d = did, d
    return best


# ---------------- main loop ----------------
for k in range(START_FRAME, len(images)):
    frame = cv2.imread(str(images[k]))
    res = model.track(frame, persist=True, tracker=tracker_cfg, classes=CLASSES,
                      conf=DET_CONF, imgsz=IMGSZ, verbose=False)[0]
    annotated, canvas = frame.copy(), bev.make_canvas(frame if show_road else None)
    hv = int(bev.horizon(pitch[k]))
    cv2.line(annotated, (0, hv), (W, hv), (0, 255, 255), 1)

    dets = []                                         # (raw tid, box, cls_id, conf, ground xz or None)
    if res.boxes.id is not None:
        for box, tid, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist(),
                                          res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
            xz = None
            if box[3] <= H - 3:                       # not cut off at the bottom
                g, ok = bev.image_to_ground([[(box[0] + box[2]) / 2, box[3]]], pitch[k])
                if ok[0]:
                    xz = g[0]
            raw_hits[tid] += 1
            dets.append((tid, box, cls_id, conf, xz))

    # resolve displayed ids: known ids first, then try to re-link brand-new ones
    busy = set()
    for tid, *_ in dets:
        if tid in alias:
            if alias[tid] in busy:                    # ByteTrack revived the old track too: split again
                alias[tid] = tid
            busy.add(alias[tid])
        else:
            busy.add(tid)                             # a young id can't be re-linked to itself
    for tid, box, cls_id, conf, xz in dets:
        if tid in alias:
            continue
        did = stitch(xz, model.names[cls_id], k, busy) if xz is not None else None
        if did is not None:
            alias[tid] = did
            print(f"frame {k}: ID {tid} re-linked to ID {did}")
        elif raw_hits[tid] >= STITCH_WINDOW or xz is None and raw_hits[tid] > 1:
            alias[tid] = tid                          # give up re-linking, keep its own id
        busy.add(alias.get(tid, tid))

    for tid, box, cls_id, conf, xz in dets:
        did = alias.get(tid, tid)
        x1, y1, x2, y2 = box
        hits[did] += 1
        best_conf[did] = max(best_conf[did], conf)
        name = model.names[stable_class(did, cls_id, conf)]
        draw = hits[did] >= MIN_HITS and best_conf[did] >= DRAW_CONF
        color = color_for_id(did)

        # BEV position (kept for every object, so re-linking also works for objects at the side)
        new = None
        if xz is not None:
            # smoothing: blend with the previous position unless it jumped (new object / bad box)
            prev = smoothed.get(did)
            new = np.array(xz)
            if prev is not None and np.linalg.norm(new - prev) < JUMP_RESET:
                new = SMOOTHING * new + (1 - SMOOTHING) * prev
            if prev is not None and did in last_seen:
                step = np.clip((new - prev) / max(k - last_seen[did], 1), -4, 4)
                velocity[did] = 0.7 * velocity.get(did, step) + 0.3 * step
            smoothed[did] = new
            trails[did].append(new)
        last_seen[did] = k
        in_bev = new is not None and abs(new[0]) <= LATERAL_LIMIT

        X, Z = xz if xz is not None else (np.nan, np.nan)
        smooth_xz = [f"{v:.3f}" for v in new] if new is not None else ["", ""]
        writer.writerow([k, did, tid, name, model.names[cls_id], f"{conf:.3f}", *np.round(box, 1),
                         f"{X:.3f}", f"{Z:.3f}", *smooth_xz])
        if not draw:
            continue

        # image box is always drawn for a confirmed track; distance only if we have one
        label = f"ID {did} {name}" + (f" {new[1]:.1f}m" if new is not None else "")
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(annotated, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if in_bev:
            bev.draw_object(canvas, did, name, list(trails[did]), color)

    cv2.putText(annotated, f"Frame {k}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    combined = np.hstack([annotated, cv2.resize(canvas, (panel_w, H), interpolation=cv2.INTER_AREA)])
    video.write(combined)
    cv2.imshow("KITTI BEV", combined)
    key = cv2.waitKey(DELAY_MS) & 0xFF
    if key == ord(" "):
        key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    if key == ord("w"):
        show_road = not show_road

video.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Saved {OUT / 'kitti_bev.mp4'} and {OUT / 'tracks_bev.csv'}")