import cv2
import mediapipe as mp
import os
import csv

# Paths
DATASET = "asl_alphabet_train"
ALLOWED = ["A", "B", "C", "D", "E" , "F", "G", "H", "I", "J", "K", "L", "M", "N" , "O", "P", "Q", "R" , "S", "T", "U", "V", "W", "X", "Y", "Z"]
CSV_FILE = "landmarks.csv"

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# CSV setup
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
    writer.writerow(header)

    for label in ALLOWED:
        folder = os.path.join(DATASET, label)

        if not os.path.isdir(folder):
            print("Missing folder:", label)
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                continue

            hand = result.multi_hand_landmarks[0]

            xs, ys = [], []
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

            writer.writerow(xs + ys + [label])

print("\nDone! CSV saved → landmarks.csv")
