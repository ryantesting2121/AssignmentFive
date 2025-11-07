from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle, json, os
from datetime import datetime

app = FastAPI()

# Load model
with open("/app/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

LOG_FILE = "/logs/prediction_logs.json"
os.makedirs("/logs", exist_ok=True)

class PredictRequest(BaseModel):
    text: str
    true_sentiment: str  # provided by user via POSTMAN or evaluate.py

@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.text
    pred = model.predict([text])[0]

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_text": text,
        "predicted_sentiment": pred,
        "true_sentiment": request.true_sentiment
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"predicted_sentiment": pred}
