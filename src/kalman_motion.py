"""Per-track constant-velocity Kalman filtering for KITTI BEV detections.

Reads outputs/tracks_bev.csv and writes outputs/tracks_kalman.csv.
Coordinates and velocities are camera-relative, NOT world-frame object motion.

Before filtering: one row per (frame, track_id) is kept (highest confidence), and
detections beyond MAX_RANGE_M are skipped. kf_reliable = 0 marks rows whose
velocity has not converged yet (its standard deviation is still above
RELIABLE_VEL_STD, e.g. in the first frames of a track or after a restart);
trajectory prediction and TTC should skip them.
"""
import csv
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / 'outputs' / 'tracks_bev.csv'
OUTPUT_CSV = ROOT / 'outputs' / 'tracks_kalman.csv'
FPS = 10.0  # Use real KITTI timestamps if available
# Measurement noise grows with distance. With a flat-ground projection, a small
# pitch error d_theta shifts depth by about Z^2 / cam_height * d_theta, so depth
# error grows with Z^2; lateral error grows roughly linearly with Z.
# Set on drive 2011_09_26 (frames 100-296) by comparing 0.5 s / 1 s prediction
# errors; this setting gave the lowest mean error at about the same median as
# looser ones, with ~3% of measurements rejected. Re-tune on other drives.
LATERAL_STD_BASE = 0.15  # metres, X error up close
LATERAL_STD_PER_M = 0.01  # extra X error per metre of distance
DEPTH_STD_BASE = 0.4  # metres, Z error up close
DEPTH_STD_PER_M2 = 0.0025  # extra Z error per metre^2 of distance (~2.6 m at 30 m)
ACCELERATION_STD = 3.0  # m/s^2; tune using validation data
INITIAL_VELOCITY_STD = 15.0  # m/s
MAX_GAP_SECONDS = 1.0  # restart tracks after long observation gaps
GATE_THRESHOLD = 13.82  # chi-square(2) 99.9% innovation gate
MAX_CONSECUTIVE_REJECTS = 3  # restart a track that keeps rejecting (the filter has drifted)
MAX_RANGE_M = 60.0  # skip detections farther than this: flat-ground depth is unusable there
RELIABLE_VEL_STD = 3.0  # m/s: velocity is trusted once the filter's own uncertainty is below this


def measurement_cov(Z):
    """2x2 measurement covariance for a detection at depth Z (metres)."""
    z = max(Z, 0.0)
    sx = LATERAL_STD_BASE + LATERAL_STD_PER_M * z
    sz = DEPTH_STD_BASE + DEPTH_STD_PER_M2 * z**2
    return np.diag([sx**2, sz**2])


class CVKalman:
    def __init__(self, X, Z):
        R = measurement_cov(Z)
        self.x = np.array([X, Z, 0.0, 0.0], dtype=float)
        self.P = np.diag([R[0, 0], R[1, 1],
                          INITIAL_VELOCITY_STD**2, INITIAL_VELOCITY_STD**2])
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.rejects = 0  # consecutive rejected measurements

    @property
    def reliable(self):
        """True once the velocity estimate has converged (it starts at 0 with a 15 m/s std)."""
        return math.sqrt(max(self.P[2, 2], self.P[3, 3])) < RELIABLE_VEL_STD

    def update(self, X, Z, dt):
        F = np.array([[1., 0., dt, 0.], [0., 1., 0., dt],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])
        G = np.array([[0.5*dt**2, 0.], [0., 0.5*dt**2], [dt, 0.], [0., dt]])
        Q = (ACCELERATION_STD**2) * (G @ G.T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        R = measurement_cov(Z)
        y = np.array([X, Z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > GATE_THRESHOLD:
            self.rejects += 1
            return 'rejected'
        self.rejects = 0
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        I = np.eye(4)
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        return 'updated'


def to_float(value):
    """CSV text -> float; blank or malformed values become NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def drop_duplicates(rows):
    """Keep one row per (frame, track_id): the highest-confidence one.

    The BEV demo's re-linking can occasionally give two boxes the same track_id in
    one frame. Two rows in one frame mean dt = 0, which would restart the filter.
    """
    best = {}
    for row in rows:
        key = (int(row['frame']), int(row['track_id']))
        conf = to_float(row.get('conf'))
        conf = conf if math.isfinite(conf) else -1.0
        if key not in best or conf > best[key][0]:
            best[key] = (conf, row)
    return [row for _, row in best.values()]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f'Missing {INPUT_CSV}; run kitti_bev_demo.py first.')
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with INPUT_CSV.open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'frame', 'track_id', 'raw_X', 'raw_Z'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required CSV columns: {required - set(reader.fieldnames or [])}')
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    n_input = len(rows)
    rows = drop_duplicates(rows)
    n_duplicates = n_input - len(rows)
    rows.sort(key=lambda r: (int(r['frame']), int(r['track_id'])))
    extra = ['kf_X', 'kf_Z', 'kf_vx', 'kf_vz', 'kf_speed', 'kf_status', 'kf_reliable']
    filters = {}
    counts = {}
    n_no_position = n_far = n_written = n_reliable = 0
    with OUTPUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra)
        writer.writeheader()
        for row in rows:
            tid, frame = int(row['track_id']), int(row['frame'])
            X, Z = to_float(row['raw_X']), to_float(row['raw_Z'])
            if not (math.isfinite(X) and math.isfinite(Z)):
                n_no_position += 1  # object was not placed in the BEV
                continue
            if Z > MAX_RANGE_M:
                n_far += 1
                continue
            previous = filters.get(tid)
            if previous is None:
                kf = CVKalman(X, Z)
                status = 'initialized'
            else:
                kf, prev_frame = previous
                dt = (frame - prev_frame) / FPS
                if dt <= 0 or dt > MAX_GAP_SECONDS:
                    kf = CVKalman(X, Z)
                    status = 'reinitialized'
                else:
                    status = kf.update(X, Z, dt)
                    if kf.rejects >= MAX_CONSECUTIVE_REJECTS:
                        kf = CVKalman(X, Z)  # filter drifted away: restart on this measurement
                        status = 'reinitialized'
            filters[tid] = (kf, frame)
            counts[status] = counts.get(status, 0) + 1
            x, z, vx, vz = kf.x
            row.update(kf_X=f'{x:.3f}', kf_Z=f'{z:.3f}',
                       kf_vx=f'{vx:.3f}', kf_vz=f'{vz:.3f}',
                       kf_speed=f'{math.hypot(vx, vz):.3f}', kf_status=status,
                       kf_reliable=int(kf.reliable))
            writer.writerow(row)
            n_written += 1
            n_reliable += kf.reliable
    print(f'Saved {OUTPUT_CSV}: {n_written} of {n_input} input rows')
    print(f'Dropped: {n_duplicates} duplicate track/frame rows, {n_no_position} without a BEV position, '
          f'{n_far} beyond {MAX_RANGE_M:.0f} m')
    print('Statuses:', counts)
    print(f'Reliable velocity (kf_reliable=1): {n_reliable} of {n_written} rows')
    print('Note: velocities are camera-relative and are not ego-motion compensated.')

if __name__ == '__main__':
    main()