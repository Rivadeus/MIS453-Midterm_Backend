import joblib
import os

class SentimentPredictor:
    def __init__(self, model_path="models/sentiment_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run src/train_model.py first.")
        self.model = joblib.load(model_path)

    def predict(self, text):
        # Predict label
        prediction = self.model.predict([text])[0]
        # Predict probability (confidence)
        proba = self.model.predict_proba([text]).max()
        
        return {
            "label": prediction,
            "confidence": round(float(proba), 2)
        }