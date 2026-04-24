from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import joblib
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        # ✅ safe check
        if "image" not in data:
            return {"error": "image missing"}

        image_str = data["image"]

        # handle both formats safely
        if "," in image_str:
            image_data = image_str.split(",")[1]
        else:
            image_data = image_str

        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "invalid image"}

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            xs, ys = [], []
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

            features = np.array(xs + ys).reshape(1, -1)

            prediction = model.predict(features)[0]

            return {"prediction": str(prediction)}

        return {"prediction": "No hand detected"}

    except Exception as e:
        return {"error": str(e)}