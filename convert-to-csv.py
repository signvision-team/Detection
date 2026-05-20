import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
DATASET  = "asl_alphabet_train"
ALLOWED  = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]
CSV_FILE = "landmarks.csv"

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (63 → 63 normalised features)
#
#   • Uses x, y, z  (21 landmarks × 3 = 63 raw values)
#   • Wrist (landmark 0) becomes the origin  → relative coords
#   • Scale-normalised by the bounding-box diagonal
#     so distance from camera does NOT matter
# ─────────────────────────────────────────────
def extract_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])      # shape (21, 3)

    coords = np.array(coords, dtype=np.float32)

    # 1. Wrist-relative
    wrist    = coords[0]
    coords  -= wrist                             # origin = wrist

    # 2. Scale normalisation
    all_xyz  = coords.flatten()
    scale    = np.max(np.abs(all_xyz))           # max absolute value
    if scale > 0:
        coords /= scale

    return coords.flatten().tolist()             # 63 values

# ─────────────────────────────────────────────
# BUILD CSV HEADER
# ─────────────────────────────────────────────
header = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    ["label"]
)

# ─────────────────────────────────────────────
# MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────
total = 0
skipped = 0

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label in ALLOWED:
        folder = os.path.join(DATASET, label)

        if not os.path.isdir(folder):
            print(f"  [SKIP] Missing folder: {label}")
            continue

        count = 0
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img  = cv2.imread(path)
            if img is None:
                skipped += 1
                continue

            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                skipped += 1
                continue

            features = extract_features(result.multi_hand_landmarks[0])
            writer.writerow(features + [label])
            count += 1
            total += 1

        print(f"  ✔ {label}: {count} images processed")

print(f"\n{'─'*45}")
print(f"  Done!  {total} rows written  →  {CSV_FILE}")
print(f"  Skipped (no hand detected): {skipped}")
print(f"  Features per sample: 63  (x,y,z × 21, wrist-relative, normalised)")
print(f"{'─'*45}")