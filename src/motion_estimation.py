
import csv
import math
from pathlib import Path

# KITTI Raw images are normally captured at 10 FPS.
# Verify that this matches your downloaded sequence.
FPS = 10.0

INPUT_CSV = Path("outputs/tracks_bev.csv")
OUTPUT_CSV = Path("outputs/tracks_motion.csv")

previous_positions = {}

with INPUT_CSV.open("r", newline="") as infile, \
     OUTPUT_CSV.open("w", newline="") as outfile:

    reader = csv.DictReader(infile)

    fieldnames = list(reader.fieldnames) + ["vx", "vz", "speed"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        track_id = int(row["track_id"])
        frame = int(row["frame"])
        if not row["X"] or not row["Z"]:
            continue
        X = float(row["X"])
        Z = float(row["Z"])

        vx = 0.0
        vz = 0.0
        speed = 0.0

        if track_id in previous_positions:
            prev_frame, prev_X, prev_Z = previous_positions[track_id]

            dt = (frame - prev_frame) / FPS

            if dt > 0:
                vx = (X - prev_X) / dt
                vz = (Z - prev_Z) / dt
                speed = math.sqrt(vx**2 + vz**2)

        previous_positions[track_id] = (frame, X, Z)

        row["vx"] = round(vx, 3)
        row["vz"] = round(vz, 3)
        row["speed"] = round(speed, 3)

        writer.writerow(row)

print(f"Motion estimation saved to {OUTPUT_CSV}")
