from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import SentimentPredictor
import os

# Initialize API
app = FastAPI(
    title="SentimentScope API",
    description="Midterm Project: Sentiment Analysis API",
    version="1.0"
)

# Input Schema
class ReviewRequest(BaseModel):
    text: str

# Global predictor variable
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    # Check if model exists, if not, try to train it or warn user
    if os.path.exists("models/sentiment_model.pkl"):
        predictor = SentimentPredictor()
    else:
        print("Warning: Model not found. Please run src/train_model.py")

@app.get("/")
def home():
    return {"message": "SentimentScope API is running. Go to /docs for interface."}

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    result = predictor.predict(request.text)
    return result