"""Per-track constant-velocity Kalman filtering for KITTI BEV detections.

Reads outputs/tracks_bev.csv and writes outputs/tracks_kalman.csv.
Coordinates and velocities are camera-relative, NOT world-frame object motion.
"""
import csv
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / 'outputs' / 'tracks_bev.csv'
OUTPUT_CSV = ROOT / 'outputs' / 'tracks_kalman.csv'
FPS = 10.0  # Use real KITTI timestamps if available
MEASUREMENT_STD = 1.0  # metres; tune using validation data
ACCELERATION_STD = 3.0  # m/s^2; tune using validation data
INITIAL_VELOCITY_STD = 15.0  # m/s
MAX_GAP_SECONDS = 1.0  # restart tracks after long observation gaps
GATE_THRESHOLD = 13.82  # chi-square(2) 99.9% innovation gate

class CVKalman:
    def __init__(self, X, Z):
        self.x = np.array([X, Z, 0.0, 0.0], dtype=float)
        self.P = np.diag([MEASUREMENT_STD**2, MEASUREMENT_STD**2,
                          INITIAL_VELOCITY_STD**2, INITIAL_VELOCITY_STD**2])
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.R = np.eye(2) * MEASUREMENT_STD**2

    def update(self, X, Z, dt):
        F = np.array([[1., 0., dt, 0.], [0., 1., 0., dt],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])
        G = np.array([[0.5*dt**2, 0.], [0., 0.5*dt**2], [dt, 0.], [0., dt]])
        Q = (ACCELERATION_STD**2) * (G @ G.T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        y = np.array([X, Z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > GATE_THRESHOLD:
            return 'rejected'
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        I = np.eye(4)
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return 'updated'


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
    rows.sort(key=lambda r: (int(r['frame']), int(r['track_id'])))
    extra = ['kf_X', 'kf_Z', 'kf_vx', 'kf_vz', 'kf_speed', 'kf_status']
    filters = {}
    counts = {}
    with OUTPUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra)
        writer.writeheader()
        for row in rows:
            tid, frame = int(row['track_id']), int(row['frame'])
            X, Z = float(row['raw_X']), float(row['raw_Z'])
            if not (math.isfinite(X) and math.isfinite(Z)):
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
            filters[tid] = (kf, frame)
            counts[status] = counts.get(status, 0) + 1
            x, z, vx, vz = kf.x
            row.update(kf_X=f'{x:.3f}', kf_Z=f'{z:.3f}',
                       kf_vx=f'{vx:.3f}', kf_vz=f'{vz:.3f}',
                       kf_speed=f'{math.hypot(vx, vz):.3f}', kf_status=status)
            writer.writerow(row)
    print(f'Saved {OUTPUT_CSV} ({len(rows)} input rows)')
    print('Statuses:', counts)
    print('Note: velocities are camera-relative and are not ego-motion compensated.')

if __name__ == '__main__':
    main()
