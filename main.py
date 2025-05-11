from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI(title="Sales Prediction API")

class AdFeatures(BaseModel):
    TV: float
    Radio: float

@app.on_event("startup")
def train_model():
    global model, scaler, model_info

    df = pd.read_csv("advertising.csv")
    df = df.drop(columns=["Newspaper"])  # drop low-relevance feature

    X = df[["TV", "Radio"]]
    y = df["Sales"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    r2 = model.score(X_test, y_test)
    model_info = {
        "intercept": model.intercept_,
        "coefficients": dict(zip(["TV", "Radio"], model.coef_)),
        "r2_score": r2
    }

@app.get("/")
def home():
    return {
        "message": "Welcome to the Improved Sales Prediction API!",
        # "model_info": model_info
    }

@app.post("/predict")
def predict_sales(features: AdFeatures):
    data = np.array([[features.TV, features.Radio]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return {"Predicted Sales": round(prediction, 2)}
