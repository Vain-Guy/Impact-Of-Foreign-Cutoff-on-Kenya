# app.py — CatBoost Prediction with Auto Feature Engineering

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Page configuration ===
st.set_page_config(
    page_title="Funding Prediction App",
    page_icon="💰",
    layout="centered"
)

# === Load historical data (for feature engineering) ===
@st.cache_data
def load_data():
    # Replace with your cleaned historical dataset
    return pd.read_csv("../Clean Data/data_no_feature_engineering.csv")  

df_history = load_data()

# === Load CatBoost pipeline ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_catboost_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'best_catboost_pipeline.pkl' is in the app folder.")
        return None

model_pipeline = load_model()

# === App Header ===
st.title("💰 U.S. Foreign Aid Funding Predictor")
st.markdown("""
Predict **future constant-dollar U.S. foreign aid allocations** using an optimized CatBoost model.  
Temporal features like rolling averages and lags are automatically calculated internally.
""")
st.divider()

# === User Input Section ===
st.subheader("📥 Input Parameters")

col1, col2 = st.columns(2)
with col1:
    fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2100, step=1, value=2025)
    is_refund = st.selectbox("Is Refund?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    managing_agency_name = st.selectbox(
        "Managing Agency",
        df_history['managing_agency_name'].unique()
    )
    funding_agency_name = st.selectbox(
        "Funding Agency",
        df_history['funding_agency_name'].unique()
    )

sector = st.selectbox(
    "Sector",
    df_history['us_sector_name'].unique()
)

st.divider()

# === Function: Compute engineered features ===
def compute_features(history_df, year, agency, fund_agency, sector):
    """
    Computes lag1, lag2, rolling mean/std, and funding growth rate for the selected entity.
    """
    df = history_df.copy()
    df_filtered = df[
        (df['managing_agency_name'] == agency) &
        (df['funding_agency_name'] == fund_agency) &
        (df['us_sector_name'] == sector)
    ].sort_values('fiscal_year')

    # Get last two years funding for lags
    lag_1 = df_filtered.loc[df_filtered['fiscal_year'] == year-1, 'constant_dollar_amount'].values
    lag_1 = lag_1[0] if len(lag_1) > 0 else 0

    lag_2 = df_filtered.loc[df_filtered['fiscal_year'] == year-2, 'constant_dollar_amount'].values
    lag_2 = lag_2[0] if len(lag_2) > 0 else 0

    # Rolling mean and std for past 3 years
    last_3 = df_filtered[df_filtered['fiscal_year'].isin([year-1, year-2, year-3])]
    rolling_mean_3yr = last_3['constant_dollar_amount'].mean() if not last_3.empty else 0
    rolling_std_3yr = last_3['constant_dollar_amount'].std() if not last_3.empty else 0

    # Funding growth rate = (last year - year before last) / lag_2
    funding_growth_rate = ((lag_1 - lag_2)/lag_2) if lag_2 != 0 else 0

    # Default transaction type
    transaction_type_name = "Grant"  # replace with actual default or most common

    return pd.DataFrame([{
        "fiscal_year": year,
        "is_refund": is_refund,
        "managing_agency_name": agency,
        "funding_agency_name": fund_agency,
        "us_sector_name": sector,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "rolling_mean_3yr": rolling_mean_3yr,
        "rolling_std_3yr": rolling_std_3yr,
        "funding_growth_rate": funding_growth_rate,
        "transaction_type_name": transaction_type_name
    }])

# === Prediction Button ===
if st.button("🔮 Predict Funding Amount"):
    if model_pipeline is not None:
        # Compute required engineered features
        input_data = compute_features(df_history, fiscal_year, managing_agency_name, funding_agency_name, sector)

        # Predict
        log_pred = model_pipeline.predict(input_data)
        prediction = np.expm1(log_pred)[0]

        st.success(f"### 💵 Predicted Constant Dollar Amount: ${prediction:,.2f}")
        st.caption("Prediction based on historical trends and agency–sector dynamics.")
    else:
        st.error("Model not loaded — unable to generate prediction.")

st.divider()
st.markdown("""
👨‍💻 *Developed by Norman Kodi*  
Powered by **CatBoost** · Deployed on **Hugging Face Spaces**
""")