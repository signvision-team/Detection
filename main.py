from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI()

# ✅ CORS (for Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later put your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model ONCE
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

# ✅ Prediction API (LANDMARKS ONLY)
@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
    
    
    import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)