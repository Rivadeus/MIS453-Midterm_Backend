# MIS453-Midterm_Backend


This is the core machine learning backend for the SentimentScope application. It exposes a REST API built with **FastAPI** that accepts text input and returns sentiment predictions (Positive/Negative) using a trained Logistic Regression model.

## Live Demo
The API is deployed and running on Hugging Face Spaces:

https://riva1205-midterm.hf.space/docs)

##Features
* **Machine Learning:** Uses TF-IDF vectorization and Logistic Regression.
* **API:** Fast and asynchronous endpoints using FastAPI.
* **CI/CD:** Automated testing and training pipeline via GitHub Actions.
* **Containerization:** Ready for deployment using Docker.

## Local Installation

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/Rivadeus/MIS453-Midterm_Backend.git](https://github.com/Rivadeus/MIS453-Midterm_Backend.git)
cd MIS453-Midterm_Backend

### Install dependencies
python -m pip install -r requirements.txt

### Train the model
This script creates the model file (models/sentiment_model.pkl) needed for predictions.
python -m src.train_model

### Start the API Server
uvicorn main:app --reload

The API will be available at: http://127.0.0.1:8000


### API
You can test the API using the automatic documentation at http://127.0.0.1:8000/docs.

Endpoint: Predict Sentiment
URL: /predict

Method: POST

Content-Type: application/json

Request Body:
{
  "text": "The movie was fantastic and the acting was great."
}

Response:
{
  "label": "Positive",
  "confidence": 0.95
}

