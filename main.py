from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque, Counter
from typing import List, Optional, Union
import numpy as np
import joblib
import time

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="SignVision API",
    description="Real-time PSL sign recognition with confidence scoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://signvision-5mwgcpa5b-wahabullahs-projects.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
try:
    model = joblib.load("model.pkl")
    print("✔ Model loaded successfully")
except Exception as e:
    print(f"✘ Model load failed: {e}")
    model = None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VOTE_WINDOW        = 10     # frames for majority voting
CONFIDENCE_THRESH  = 0.60   # minimum confidence to accept prediction
SENTENCE_DELAY     = 2.0    # seconds before appending sign to sentence

# ─────────────────────────────────────────────
# STATE (In-Memory Tracking)
# ─────────────────────────────────────────────
prediction_buffer  = deque(maxlen=VOTE_WINDOW)
sentence_buffer    : List[str] = []

last_stable_sign   : Optional[str] = None
last_sign_time     : float = 0.0

# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]          # 63 normalised x,y,z landmarks

class LifecycleRequest(BaseModel):
    user_id: Union[str, int]       
    lesson_id: Optional[Union[str, int]] = None  

# ─────────────────────────────────────────────
# HELPER — majority vote over recent frames
# ─────────────────────────────────────────────
def majority_vote(buf: deque) -> Optional[str]:
    if not buf:
        return None
    most_common, count = Counter(buf).most_common(1)[0]
    if count / len(buf) >= 0.6:
        return most_common
    return None

# ─────────────────────────────────────────────
# STANDARD ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message"  : "SignVision API v2.0 🚀",
        "status"   : "running",
        "endpoints": ["/predict", "/sentence", "/sentence/clear", "/health", "/api/detection"]
    }


@app.get("/health")
def health():
    return {
        "model_loaded"     : model is not None,
        "buffer_size"      : len(prediction_buffer),
        "sentence_length"  : len(sentence_buffer),
    }


@app.post("/predict")
def predict(data: PredictRequest):
    global last_stable_sign, last_sign_time

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array(data.features).reshape(1, -1)

        # ── Raw prediction + confidence ──────────────────
        raw_pred   = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        confidence = float(np.max(proba))

        # ── Push to voting buffer ─────────────────────────
        if confidence >= CONFIDENCE_THRESH:
            prediction_buffer.append(raw_pred)
        else:
            prediction_buffer.append(None)   

        # ── Majority vote for stable sign ─────────────────
        stable_sign = majority_vote(prediction_buffer)

        # ── Sentence buffer logic ─────────────────────────
        now = time.time()
        appended_to_sentence = False

        if (stable_sign is not None
                and stable_sign != last_stable_sign
                and confidence >= CONFIDENCE_THRESH):
            last_stable_sign = stable_sign
            last_sign_time   = now

        elif (stable_sign is not None
                and stable_sign == last_stable_sign
                and (now - last_sign_time) >= SENTENCE_DELAY
                and confidence >= CONFIDENCE_THRESH):
            sentence_buffer.append(stable_sign)
            last_sign_time      = now 
            appended_to_sentence = True

        return {
            "raw_prediction"       : str(raw_pred),
            "confidence"           : round(confidence, 4),
            "stable_sign"          : stable_sign,
            "appended_to_sentence" : appended_to_sentence,
            # FIXED: Added space character separator for word readability
            "sentence"             : " ".join(sentence_buffer),
            "buffer_votes"         : list(prediction_buffer),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentence")
def get_sentence():
    return {
        "sentence"      : " ".join(sentence_buffer),
        "signs_captured": len(sentence_buffer)
    }


@app.post("/sentence/clear")
def clear_sentence():
    global sentence_buffer, last_stable_sign, last_sign_time
    sentence_buffer  = []
    last_stable_sign = None
    last_sign_time   = 0.0
    prediction_buffer.clear()
    return {"message": "Sentence cleared ✔", "sentence": ""}


@app.post("/sentence/backspace")
def backspace():
    if sentence_buffer:
        sentence_buffer.pop()
    return {"sentence": " ".join(sentence_buffer)}


# ─────────────────────────────────────────────────────────────
# LIFECYCLE DETECTION ENDPOINTS FOR COMPONENT COUPLING
# ─────────────────────────────────────────────────────────────

@app.post("/api/detection/start")
def start_detection(payload: LifecycleRequest):
    global sentence_buffer, last_stable_sign, last_sign_time
    sentence_buffer = []
    last_stable_sign = None
    last_sign_time = 0.0
    prediction_buffer.clear()
    
    print(f"🚀 Tracking pipeline started for user: {payload.user_id}")
    return {"status": "started", "user_id": payload.user_id}


@app.get("/api/detection/current/{user_id}")
def get_current_detection(user_id: str):
    current_stable = majority_vote(prediction_buffer)
    
    prediction_payload = ""
    if current_stable:
        prediction_payload = str(current_stable)
    elif len(prediction_buffer) > 0 and prediction_buffer[-1] is not None:
        prediction_payload = str(prediction_buffer[-1])

    return {
        "user_id": user_id,
        "prediction": prediction_payload,
        "stable_sign": current_stable,
        "sentence": " ".join(sentence_buffer)
    }


@app.post("/api/detection/stop")
def stop_detection(payload: LifecycleRequest):
    global sentence_buffer, last_stable_sign, last_sign_time
    sentence_buffer = []
    last_stable_sign = None
    last_sign_time = 0.0
    prediction_buffer.clear()
    
    print(f"🛑 Tracking pipeline stopped for user: {payload.user_id}")
    return {"status": "stopped", "user_id": payload.user_id}