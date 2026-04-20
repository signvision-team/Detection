import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained SVM (42-feature model)
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            xs, ys = [], []
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

            row = np.array(xs + ys).reshape(1, -1)

            pred = model.predict(row)[0]

            cv2.putText(img, f"Detected: {pred}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("PSL Sign Detector", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
