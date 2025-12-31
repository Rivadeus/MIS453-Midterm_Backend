import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Import from our new files
from src.data_loader import load_data
from src.preprocess import clean_text

os.makedirs("models", exist_ok=True)

def train():
    print("Loading data...")
    df = load_data()
    
    # Apply preprocessing
    df['text'] = df['text'].apply(clean_text)
    
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Save
    joblib.dump(pipeline, "models/sentiment_model.pkl")
    print("Model saved to models/sentiment_model.pkl")

if __name__ == "__main__":
    train()