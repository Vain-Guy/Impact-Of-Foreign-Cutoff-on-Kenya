# app.py - FINAL CORRECTED VERSION

# 1. Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np 

# === Initialize app ===
app = FastAPI(
    title="Funding Prediction API",
    description="API that predicts U.S. foreign aid funding amounts using an optimized XGBoost pipeline.",
    version="1.0"
)

# === Load trained model pipeline ===
# This pipeline includes your preprocessor and the XGBoost model
try:
    model_pipeline = joblib.load("best_xgb_pipeline.pkl")
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: best_xgb_pipeline.pkl not found. Make sure it's in the App folder.")
    model_pipeline = None

# === Define input schema ===
# These fields MUST match the columns your pipeline was trained on.
class FundingInput(BaseModel):
    fiscal_year: int
    is_refund: int
    managing_agency_name: str
    funding_agency_name: str
    sector: str  
    
    # These are your engineered features. The user of the API
    # would need to provide these. We can give them default values.
    lag_1: float = 0
    lag_2: float = 0
    rolling_mean_3yr: float = 0
    rolling_std_3yr: float = 0
    funding_growth_rate: float = 0

# === Define prediction endpoint ===
@app.post("/predict")
def predict_funding(data: FundingInput):
    """
    Predicts funding amount given input features.
    """
    if not model_pipeline:
        return {"error": "Model is not loaded."}

    # Convert the Pydantic model to a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # The pipeline will handle all preprocessing (OneHotEncoding, etc.)
    # and then make a prediction.
    log_prediction = model_pipeline.predict(input_df)
    
    # CRITICAL: Reverse the log-transformation to get the actual dollar amount
    prediction_amount = np.expm1(log_prediction)[0]
    
    return {
        "predicted_constant_dollar_amount": float(prediction_amount)
    }

# === Root endpoint ===
@app.get("/")
def home():
    return {"message": "Welcome to the Funding Prediction API! Visit /docs for the interactive Swagger UI."}