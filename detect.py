import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
from typing import Optional

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = joblib.load("model.pkl")

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VOTE_WINDOW        = 10
CONFIDENCE_THRESH  = 0.60
SENTENCE_DELAY_FPS = 60      # ~2 seconds at 30 fps → use frame counter

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
prediction_buffer  = deque(maxlen=VOTE_WINDOW)
sentence           = []
last_stable_sign   : Optional[str] = None
stable_hold_frames : int = 0

# ─────────────────────────────────────────────
# FEATURE EXTRACTION  (matches convert_to_csv.py)
# ─────────────────────────────────────────────
def extract_features(hand_landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                      dtype=np.float32)
    coords -= coords[0]                          # wrist-relative
    scale   = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale
    return coords.flatten().reshape(1, -1)       # (1, 63)

# ─────────────────────────────────────────────
# MAJORITY VOTE
# ─────────────────────────────────────────────
def majority_vote(buf: deque) -> Optional[str]:
    filtered = [x for x in buf if x is not None]
    if not filtered:
        return None
    most_common, count = Counter(filtered).most_common(1)[0]
    if count / len(buf) >= 0.6:
        return most_common
    return None

# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def confidence_bar(img, conf, x, y, w=200, h=16):
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,60,60), -1)
    fill  = int(w * conf)
    color = (0,220,100) if conf >= 0.80 else (0,180,220) if conf >= 0.60 else (0,80,220)
    cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (180,180,180), 1)
    cv2.putText(img, f"{conf*100:.0f}%", (x+w+8, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("  Press  Q  to quit   |   SPACE  to clear sentence   |   BACKSPACE  to delete last")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_sign : Optional[str] = None
    confidence   : float         = 0.0
    stable_sign  : Optional[str] = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # draw skeleton
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(80,200,255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(255,180,80), thickness=2))

        # predict
        features   = extract_features(hand)
        proba      = model.predict_proba(features)[0]
        confidence = float(np.max(proba))
        current_sign = model.predict(features)[0]

        # buffer
        if confidence >= CONFIDENCE_THRESH:
            prediction_buffer.append(current_sign)
        else:
            prediction_buffer.append(None)

        stable_sign = majority_vote(prediction_buffer)

        # sentence logic
        if stable_sign is not None:
            if stable_sign != last_stable_sign:
                last_stable_sign   = stable_sign
                stable_hold_frames = 0
            else:
                stable_hold_frames += 1
                if stable_hold_frames == SENTENCE_DELAY_FPS:
                    sentence.append(stable_sign)
        else:
            stable_hold_frames = 0

    else:
        prediction_buffer.clear()
        last_stable_sign   = None
        stable_hold_frames = 0

    # ── OVERLAY UI ───────────────────────────────────────
    # Top panel
    draw_rounded_rect(img, 10, 10, 340, 110, 10, (30,30,30))

    if current_sign:
        cv2.putText(img, current_sign, (24, 72),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, (80,220,255), 3)
        cv2.putText(img, "raw prediction", (24, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        confidence_bar(img, confidence, 110, 50, w=200)
    else:
        cv2.putText(img, "No hand", (24, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,100), 2)

    # Stable sign badge
    if stable_sign:
        color = (0,220,100) if stable_hold_frames >= SENTENCE_DELAY_FPS else (0,200,255)
        cv2.putText(img, f"Stable: {stable_sign}", (24, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Hold progress bar
    if stable_sign and stable_hold_frames < SENTENCE_DELAY_FPS:
        pct = stable_hold_frames / SENTENCE_DELAY_FPS
        bw  = 300
        cv2.rectangle(img, (10, 125), (10+bw, 135), (50,50,50), -1)
        cv2.rectangle(img, (10, 125), (10+int(bw*pct), 135), (0,200,120), -1)
        cv2.putText(img, "Hold to add …", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

    # Sentence panel (bottom)
    sentence_str = "".join(sentence)
    draw_rounded_rect(img, 10, h-60, w-20, 50, 8, (20,20,20))
    cv2.putText(img, f"Sentence: {sentence_str}", (22, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255,255,255) if sentence_str else (90,90,90), 2)

    # Controls hint
    cv2.putText(img, "Q: quit   SPACE: clear   BKSP: delete",
                (10, h-68), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120,120,120), 1)

    cv2.imshow("SignVision — Real-Time PSL Detector", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        sentence.clear()
        print("  Sentence cleared")
    elif key == 8:    # backspace
        if sentence:
            sentence.pop()

cap.release()
cv2.destroyAllWindows()