"""
Bird's-eye-view (BEV) projection for KITTI with a flat-ground homography.

Frame: rectified camera, X = right, Z = forward (metres), road at Y = +1.65 m.
A road point [X, h, Z] projects as p ~ P2 @ [X, h, Z, 1], which collapses to a
3x3 homography  H = [P2[:,0] | P2[:,2] | h*P2[:,1] + P2[:,3]].  inv(H) maps a
pixel on the road back to metres.
"""

from pathlib import Path

import cv2
import numpy as np

CAMERA_HEIGHT = 1.65  # m

# typical footprint (width, length) in metres, for drawing
OBJECT_SIZE_M = {"car": (1.8, 4.5), "truck": (2.5, 8.0), "bus": (2.6, 11.0),
                 "person": (0.6, 0.6), "bicycle": (0.6, 1.8), "motorcycle": (0.8, 2.1)}



def load_p2(calib_path):
    """3x4 projection matrix for KITTI's left grayscale camera (image_00)."""
    with open(calib_path) as f:
        for line in f:
            key, _, vals = line.partition(":")
            if key.strip() == "P_rect_00":
                return np.array(vals.split(), float).reshape(3, 4)
    raise ValueError(f"No P_rect_00 in {calib_path}")


def camera_pitch_from_calib(path):
    """Static camera pitch vs the car body (rad, + = nose down), via camera -> LiDAR -> IMU.

    path: a raw-data calib folder (calib_cam_to_cam / velo_to_cam / imu_to_velo .txt)
          or a single tracking-benchmark calib file (R_rect, Tr_velo_cam, Tr_imu_velo).
    """
    def read(f):
        return {k.strip(): np.array(v.split(), float) for k, _, v in
                (l.partition(":") for l in open(f)) if v.strip() and k.strip() != "calib_time"}

    def rt(R, T):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R.reshape(3, 3), T
        return M

    p = Path(path)
    if p.is_dir():
        R_rect = read(p / "calib_cam_to_cam.txt")["R_rect_00"]
        vc, iv = read(p / "calib_velo_to_cam.txt"), read(p / "calib_imu_to_velo.txt")
        velo_cam, imu_velo = rt(vc["R"], vc["T"]), rt(iv["R"], iv["T"])
    else:
        c = read(p)
        R_rect = c["R_rect"]
        velo_cam, imu_velo = np.vstack([c["Tr_velo_cam"].reshape(3, 4), [0, 0, 0, 1]]), \
                             np.vstack([c["Tr_imu_velo"].reshape(3, 4), [0, 0, 0, 1]])
    cam_from_imu = rt(R_rect, np.zeros(3)) @ velo_cam @ imu_velo
    forward_in_imu = cam_from_imu[:3, :3].T @ [0, 0, 1]   # IMU axes: x fwd, y left, z up
    return -np.arcsin(forward_in_imu[2])


def load_pitch(oxts_dir, n_frames, window=11):
    """Per-frame camera pitch wobble (rad, + = nose down) from KITTI raw OXTS.

    Only the fast part is kept: slow pitch changes are hills (the road tilts with
    the car, so the flat-ground model still holds); fast changes are the car
    bouncing/braking, which shifts every projected distance.
    Returns zeros if the OXTS folder is missing.
    """
    files = sorted(Path(oxts_dir, "data").glob("*.txt"))
    if not files:
        print(f"No OXTS in {oxts_dir} - pitch correction off")
        return np.zeros(n_frames)
    pitch = np.array([float(open(f).read().split()[4]) for f in files])
    smooth = np.convolve(np.pad(pitch, window // 2, mode="edge"), np.ones(window) / window, "valid")
    out = np.zeros(n_frames)
    m = min(n_frames, len(pitch))
    out[:m] = (pitch - smooth)[:m]
    return out


class GroundPlaneBEV:
    def __init__(self, P2, cam_height=CAMERA_HEIGHT, x_range=(-15, 15), z_range=(0, 60), px_per_m=10,
                 static_pitch=0.0):
        """static_pitch: fixed camera tilt (rad, + = nose down), e.g. from camera_pitch_from_calib."""
        self.P2, self.h = np.asarray(P2, float), cam_height
        self.static_pitch = static_pitch
        self.x_min, self.x_max = x_range
        self.z_min, self.z_max = z_range
        self.ppm = px_per_m
        self.bev_w = int((self.x_max - self.x_min) * px_per_m)
        self.bev_h = int((self.z_max - self.z_min) * px_per_m)
        self.fy, self.cy = self.P2[1, 1], self.P2[1, 2]
        self.horizon_v = self.horizon(0.0)
        # image -> BEV canvas, for warping the road picture
        A = np.array([[px_per_m, 0, -self.x_min * px_per_m], [0, -px_per_m, self.z_max * px_per_m], [0, 0, 1]])
        self.H_bev_from_img = A @ self._H_ground_from_img(0.0)

    def _H_ground_from_img(self, pitch):
        pitch = pitch + self.static_pitch
        c, s = np.cos(pitch), np.sin(pitch)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # level frame -> pitched camera
        P = np.hstack([self.P2[:, :3] @ R, self.P2[:, 3:]])
        return np.linalg.inv(np.column_stack([P[:, 0], P[:, 2], P[:, 1] * self.h + P[:, 3]]))

    def horizon(self, pitch=0.0):
        """Image row of the horizon; nothing at/above it can be on the road."""
        return self.cy - self.fy * np.tan(pitch + self.static_pitch)

    def image_to_ground(self, pts, pitch=0.0):
        """Pixels (N,2) -> ground (N,2) [X, Z] metres, plus a validity mask."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        g = np.hstack([pts, np.ones((len(pts), 1))]) @ self._H_ground_from_img(pitch).T
        with np.errstate(divide="ignore", invalid="ignore"):
            xz = g[:, :2] / g[:, 2:3]
        valid = (pts[:, 1] > self.horizon(pitch) + 3) & (xz[:, 1] > 0.5)
        return xz, valid

    def ground_to_bev(self, xz):
        xz = np.asarray(xz, float).reshape(-1, 2)
        return np.stack([(xz[:, 0] - self.x_min) * self.ppm, (self.z_max - xz[:, 1]) * self.ppm], axis=1)

    def in_map(self, xz):
        x, z = xz
        return self.x_min <= x <= self.x_max and self.z_min <= z <= self.z_max

    # ---------------- drawing ----------------

    def make_canvas(self, frame=None):
        """Dark canvas (or the warped road) with a metric grid and the ego car."""
        if frame is not None:
            canvas = (cv2.warpPerspective(frame, self.H_bev_from_img, (self.bev_w, self.bev_h)) * 0.55).astype(np.uint8)
        else:
            canvas = np.zeros((self.bev_h, self.bev_w, 3), np.uint8)
        for z in range(10, int(self.z_max) + 1, 10):
            r = int((self.z_max - z) * self.ppm)
            cv2.line(canvas, (0, r), (self.bev_w, r), (90, 90, 90), 1)
            cv2.putText(canvas, f"{z} m", (4, r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        for x in range(int(self.x_min), int(self.x_max) + 1, 5):
            c = int((x - self.x_min) * self.ppm)
            cv2.line(canvas, (c, 0), (c, self.bev_h), (140, 140, 140) if x == 0 else (90, 90, 90), 1)
        cx = int(-self.x_min * self.ppm)
        cv2.fillPoly(canvas, [np.array([[cx, self.bev_h - 16], [cx - 7, self.bev_h - 2], [cx + 7, self.bev_h - 2]])],
                     (255, 255, 255))
        return canvas

    def draw_object(self, canvas, track_id, cls_name, trail_xz, color):
        """Real-size footprint + trail + 'ID distance' label for one object."""
        trail = np.asarray(trail_xz, float).reshape(-1, 2)
        foot = trail[-1]
        if not self.in_map(foot):
            return
        # the box bottom is the object's NEAREST point: push the footprint
        # half a length away from the camera along the viewing ray
        w, l = OBJECT_SIZE_M.get(cls_name, (1.5, 3.0))
        ray = foot / max(np.linalg.norm(foot), 1e-6)
        c = foot + ray * (l / 2 * abs(ray[1]) + w / 2 * abs(ray[0]))
        corners = c + np.array([[w, l], [-w, l], [-w, -l], [w, -l]]) / 2
        fp = self.ground_to_bev(corners).astype(np.int32)

        if len(trail) > 1:
            cv2.polylines(canvas, [self.ground_to_bev(trail).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [fp], color)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)
        cv2.polylines(canvas, [fp], True, color, 2, cv2.LINE_AA)
        label = f"{track_id} {foot[1]:.0f}m"
        org = (int(fp[:, 0].max()) + 4, int(fp[:, 1].mean()) + 4)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)