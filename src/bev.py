"""
Bird's-eye-view (BEV) projection for KITTI using a flat-ground homography
(inverse perspective mapping derived from the KITTI calibration, no manual
point picking).

Coordinate frame = KITTI rectified left colour camera (cam2):
    X : metres to the right of the camera
    Y : metres DOWN (the road surface is at Y = +CAMERA_HEIGHT)
    Z : metres forward

Core idea:
    A ground point [X, h, Z] projects to the image as  p ~ P2 @ [X, h, Z, 1].
    Because Y is fixed at h, that collapses to a 3x3 homography:
        p ~ H @ [X, Z, 1],   H = [P2[:,0] | P2[:,2] | h*P2[:,1] + P2[:,3]]
    Inverting H maps any image pixel that lies on the road back to metres.
"""

import cv2
import numpy as np

CAMERA_HEIGHT = 1.65  # metres; KITTI cameras sit 1.65 m above the road


def load_p2(calib_path):
    """Read the 3x4 projection matrix of the left colour camera.

    Handles tracking/object calib files ("P2: ...") and raw-data
    calib_cam_to_cam.txt ("P_rect_02: ...").
    """
    with open(calib_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            if key.strip() in ("P2", "P_rect_02"):
                return np.array(values.split(), dtype=np.float64).reshape(3, 4)
    raise ValueError(f"No P2 / P_rect_02 entry found in {calib_path}")


class GroundPlaneBEV:
    def __init__(
        self,
        P2,
        cam_height=CAMERA_HEIGHT,
        x_range=(-20.0, 20.0),   # lateral extent of the BEV map (m)
        z_range=(0.0, 50.0),     # forward extent of the BEV map (m)
        px_per_m=8,
    ):
        self.P2 = np.asarray(P2, dtype=np.float64)
        self.h = cam_height
        self.x_min, self.x_max = x_range
        self.z_min, self.z_max = z_range
        self.ppm = px_per_m

        # ground (X, Z, 1) -> image pixel (homogeneous)
        self.H_img_from_ground = np.column_stack(
            [self.P2[:, 0], self.P2[:, 2], self.P2[:, 1] * cam_height + self.P2[:, 3]]
        )
        self.H_ground_from_img = np.linalg.inv(self.H_img_from_ground)

        # ground (X, Z) -> BEV canvas pixel (col, row), forward = up
        self.A_bev_from_ground = np.array(
            [
                [self.ppm, 0.0, -self.x_min * self.ppm],
                [0.0, -self.ppm, self.z_max * self.ppm],
                [0.0, 0.0, 1.0],
            ]
        )
        self.H_bev_from_img = self.A_bev_from_ground @ self.H_ground_from_img

        self.bev_w = int(round((self.x_max - self.x_min) * self.ppm))
        self.bev_h = int(round((self.z_max - self.z_min) * self.ppm))

        # Horizon row = vanishing point of the forward (Z) direction.
        # Nothing at or above this row can be on the road.
        vp = self.H_img_from_ground[:, 1]
        self.horizon_v = vp[1] / vp[2]
        self.fy = self.P2[1, 1]

    # ---------------- point projections ----------------

    def image_to_ground(self, pts, horizon_margin_px=3.0, min_z=0.5):
        """Image pixels (N,2) -> ground (N,2) metres [X, Z], plus a validity mask."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        g = homo @ self.H_ground_from_img.T
        with np.errstate(divide="ignore", invalid="ignore"):
            xz = g[:, :2] / g[:, 2:3]
        valid = (pts[:, 1] > self.horizon_v + horizon_margin_px) & (xz[:, 1] > min_z)
        return xz, valid

    def ground_to_image(self, xz):
        """Ground (N,2) [X, Z] metres -> image pixels (N,2)."""
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        homo = np.hstack([xz, np.ones((len(xz), 1))])
        p = homo @ self.H_img_from_ground.T
        return p[:, :2] / p[:, 2:3]

    def ground_to_bev(self, xz):
        """Ground (N,2) [X, Z] metres -> BEV canvas pixels (N,2) [col, row]."""
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        col = (xz[:, 0] - self.x_min) * self.ppm
        row = (self.z_max - xz[:, 1]) * self.ppm
        return np.stack([col, row], axis=1)

    def in_map(self, xz):
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        return (
            (xz[:, 0] >= self.x_min) & (xz[:, 0] <= self.x_max)
            & (xz[:, 1] >= self.z_min) & (xz[:, 1] <= self.z_max)
        )

    def depth_error_per_pixel(self, z):
        """Approx. metres of depth error per 1 px of foot-point error at distance z.

        dZ/dv ~= Z^2 / (fy * h). Grows quadratically - this is the main
        accuracy limit of monocular flat-ground BEV.
        """
        return z ** 2 / (self.fy * self.h)

    # ---------------- visualisation ----------------

    def warp(self, frame):
        """Warp the whole camera image into the top-down view."""
        return cv2.warpPerspective(
            frame, self.H_bev_from_img, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR
        )

    def make_canvas(self, frame=None, road_alpha=0.55):
        """BEV canvas with the warped road (optional), a metric grid and the ego car."""
        if frame is not None:
            canvas = (self.warp(frame).astype(np.float32) * road_alpha).astype(np.uint8)
        else:
            canvas = np.zeros((self.bev_h, self.bev_w, 3), np.uint8)

        grid = (90, 90, 90)
        for z in range(10, int(self.z_max) + 1, 10):
            r = int((self.z_max - z) * self.ppm)
            cv2.line(canvas, (0, r), (self.bev_w, r), grid, 1)
            cv2.putText(canvas, f"{z} m", (4, r - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)
        x = int(np.ceil(self.x_min / 5.0) * 5)
        while x <= self.x_max:
            c = int((x - self.x_min) * self.ppm)
            cv2.line(canvas, (c, 0), (c, self.bev_h), (140, 140, 140) if x == 0 else grid, 1)
            x += 5

        # ego vehicle marker at the bottom centre
        cx = int(-self.x_min * self.ppm)
        cy = self.bev_h - 2
        tri = np.array([[cx, cy - 14], [cx - 7, cy], [cx + 7, cy]], np.int32)
        cv2.fillPoly(canvas, [tri], (255, 255, 255))
        return canvas

    # ---------------- tracked objects on the map ----------------

    def object_footprint(self, foot_xz, cls_name, heading=None):
        """Ground-plane rectangle (4x2, metres) for a detected object.

        The box's bottom-centre is the object's *nearest visible* point, not its
        centre, so the rectangle is pushed away from the camera by half its
        extent along the viewing ray.
        heading: unit (dx, dz) direction of travel; defaults to along the road (+Z).
        """
        w, l = OBJECT_SIZE_M.get(cls_name, DEFAULT_SIZE_M)
        h = np.array([0.0, 1.0]) if heading is None else np.asarray(heading, float)
        n = np.array([h[1], -h[0]])  # perpendicular (sideways) axis
        foot = np.asarray(foot_xz, float)
        r = foot / max(np.linalg.norm(foot), 1e-6)  # viewing ray on the ground
        half_depth = (l / 2) * abs(r @ h) + (w / 2) * abs(r @ n)
        c = foot + r * half_depth
        return np.array([
            c + h * l / 2 + n * w / 2,
            c + h * l / 2 - n * w / 2,
            c - h * l / 2 - n * w / 2,
            c - h * l / 2 + n * w / 2,
        ])

    def draw_object(self, canvas, track_id, cls_name, history_xz, color):
        """Draw one tracked object on the BEV canvas: footprint, trail, label.

        history_xz: (N,2) ground positions of this track, oldest first, last = now.
        Returns True if it was drawn inside the map, False if out of range.
        """
        hist = np.asarray(history_xz, float).reshape(-1, 2)
        now = hist[-1]

        # Heading from recent motion if the track has moved enough, else along road.
        # (Positions are ego-relative, so this is relative motion - fine for drawing.)
        heading = None
        if len(hist) >= 5:
            d = hist[-1] - hist[-5]
            if np.linalg.norm(d) > 1.5:
                heading = d / np.linalg.norm(d)
                if heading[1] < 0:  # keep a consistent orientation for the rectangle
                    heading = -heading

        if not self.in_map(now)[0]:
            self._draw_out_of_range(canvas, track_id, now, color)
            return False

        # trail
        trail = self.ground_to_bev(hist).astype(np.int32)
        for k in range(1, len(trail)):
            cv2.line(canvas, tuple(trail[k - 1]), tuple(trail[k]), color, 1, cv2.LINE_AA)

        # footprint box
        fp = self.ground_to_bev(self.object_footprint(now, cls_name, heading)).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [fp], color)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)
        cv2.polylines(canvas, [fp], True, color, 2, cv2.LINE_AA)

        # label: ID + range
        c, r = fp.mean(axis=0).astype(int)
        label = f"{track_id} {now[1]:.0f}m"
        cv2.putText(canvas, label, (c + 8, r + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (c + 8, r + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
        return True

    def _draw_out_of_range(self, canvas, track_id, xz, color):
        """Edge marker for a track that exists but lies beyond the map extent."""
        x = np.clip(xz[0], self.x_min + 0.5, self.x_max - 0.5)
        z = np.clip(xz[1], self.z_min + 0.5, self.z_max - 0.5)
        c, r = self.ground_to_bev([[x, z]])[0].astype(int)
        tri = np.array([[c, r - 6], [c - 5, r + 4], [c + 5, r + 4]], np.int32)
        cv2.fillPoly(canvas, [tri], color)
        cv2.putText(canvas, f"{track_id} {xz[1]:.0f}m", (c + 7, r + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


# Typical footprint (width, length) in metres per COCO class name
OBJECT_SIZE_M = {
    "car": (1.8, 4.5),
    "truck": (2.5, 8.0),
    "bus": (2.6, 11.0),
    "person": (0.6, 0.6),
    "bicycle": (0.6, 1.8),
    "motorcycle": (0.8, 2.1),
}
DEFAULT_SIZE_M = (1.5, 3.0)