from fastapi import FastAPI
import base64
import numpy as np
import cv2
import joblib
import mediapipe as mp

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/detect")
def detect(data: dict):
    try:
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            xs, ys = [], []
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

            row = np.array(xs + ys).reshape(1, -1)
            pred = model.predict(row)[0]

            return {"prediction": pred}

        return {"prediction": "No hand detected"}

    except Exception as e:
        return {"error": str(e)}
    
    
    import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)