from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI(title="Iris Classification API")

# Load model
model = joblib.load("models/model.pkl")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "Iris Classification API is Running!"
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]
    ])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    return {
        "prediction": int(prediction),
        "probabilities": probability.tolist()
    }