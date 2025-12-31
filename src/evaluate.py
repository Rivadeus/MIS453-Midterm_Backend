import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.preprocess import clean_text

def evaluate_model():
    model_path = "models/sentiment_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py first.")
        return

    # Load model
    pipeline = joblib.load(model_path)

    # Load fresh data (in real life, load a separate test set)
    df = load_data()
    
    # Preprocess
    df['text'] = df['text'].apply(clean_text)
    
    X = df['text']
    y = df['label']

    print("Running evaluation...")
    preds = pipeline.predict(X)
    
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc:.2f}")
    print("\nDetailed Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model()