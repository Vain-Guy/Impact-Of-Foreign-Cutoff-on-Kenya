import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="U.S. Foreign Aid Funding Predictor",
    page_icon="💰",
    layout="centered",
)

# === LOAD HISTORICAL DATA ===
@st.cache_data
def load_data():
    path = "modeling data.csv"
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}. Make sure it’s uploaded or in the same directory.")
        st.stop()
    return pd.read_csv(path)

# --- Check if the environment has the capability to run the app (simulating load) ---
# NOTE: In a real environment, the app would stop here if the file is missing.
try:
    df_history = load_data()
except Exception:
    df_history = None # Allow the rest of the app to render with appropriate errors

# === LOAD XGBOOST PIPELINE ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_xgb_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Ensure 'best_xgb_pipeline.pkl' is in the app folder.")
        return None

model_pipeline = load_model()

# === HEADER ===
st.title("💰 U.S. Foreign Aid Funding Predictor")
st.markdown("""
Forecast **future U.S. foreign aid allocations** using an optimized **XGBoost machine learning model**. 
All required temporal and historical features are computed automatically.
""")
st.divider()

# Only proceed with inputs and prediction if data is available for generating lists
if df_history is not None:
    # === INPUT SECTION ===
    st.header("Inputs")

    with st.expander("Set Prediction Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fiscal_year = st.number_input("Fiscal Year", 2000, 2100, 2025, 1)
            is_refund = st.radio("Refund Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            managing_agency_name = st.selectbox(
                "Managing Agency", 
                sorted(df_history['managing_agency_name'].unique())
            )
            funding_agency_name = st.selectbox(
                "Funding Agency", 
                sorted(df_history['funding_agency_name'].unique())
            )

        sector = st.selectbox("Sector", sorted(df_history['sector'].unique()))

    st.divider()

    # === FEATURE ENGINEERING FUNCTION ===
    def compute_features(history_df, year, agency, fund_agency, sector, is_refund):
        df_filtered = history_df[
            (history_df['managing_agency_name'] == agency) &
            (history_df['funding_agency_name'] == fund_agency) &
            (history_df['sector'] == sector)
        ].sort_values('fiscal_year')

        # Lag features
        lag_1 = df_filtered.loc[df_filtered['fiscal_year'] == year-1, 'constant_dollar_amount'].values
        lag_1 = lag_1[0] if len(lag_1) > 0 else 0

        lag_2 = df_filtered.loc[df_filtered['fiscal_year'] == year-2, 'constant_dollar_amount'].values
        lag_2 = lag_2[0] if len(lag_2) > 0 else 0

        # Rolling statistics
        last_3 = df_filtered[df_filtered['fiscal_year'].isin([year-1, year-2, year-3])]
        rolling_mean_3yr = last_3['constant_dollar_amount'].mean() if not last_3.empty else 0
        rolling_std_3yr = last_3['constant_dollar_amount'].std() if not last_3.empty else 0

        # Growth rate
        funding_growth_rate = ((lag_1 - lag_2) / lag_2) if lag_2 != 0 else 0

        # Transaction type
        # Mode is generally safer for categorical imputation, use 'Grant' as an absolute fallback
        transaction_type_name = df_filtered['transaction_type_name'].mode().values[0] if not df_filtered.empty and not df_filtered['transaction_type_name'].mode().empty else "Grant"

        return pd.DataFrame([{
            "fiscal_year": year,
            "is_refund": is_refund,
            "managing_agency_name": agency,
            "funding_agency_name": fund_agency,
            "sector": sector,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "rolling_mean_3yr": rolling_mean_3yr,
            "rolling_std_3yr": rolling_std_3yr,
            "funding_growth_rate": funding_growth_rate,
            "transaction_type_name": transaction_type_name
        }])

    # === HISTORICAL TREND VISUALIZATION ===
    df_plot = df_history[
        (df_history['managing_agency_name'] == managing_agency_name) &
        (df_history['funding_agency_name'] == funding_agency_name) &
        (df_history['sector'] == sector)
    ].sort_values('fiscal_year')

    sector_avg = (
        df_history[df_history['sector'] == sector]
        .groupby('fiscal_year')['constant_dollar_amount']
        .mean()
        .reset_index()
    )

    if not df_plot.empty:
        st.subheader("Historical Funding Trend")
        st.markdown(
            f"<p style='font-size:13px;color:gray;'>Comparing <b>{sector}</b> funding trajectory vs. sector-wide average.</p>",
            unsafe_allow_html=True
        )

        base = alt.Chart(df_plot).mark_line(point=True, color='#0077b6').encode(
            x='fiscal_year:O',
            y=alt.Y('constant_dollar_amount:Q', title='Funding (USD)'),
            tooltip=[alt.Tooltip('fiscal_year', title='Year'), alt.Tooltip('constant_dollar_amount', title='Amount', format='$,.2f')]
        )

        avg_line = alt.Chart(sector_avg).mark_line(
            strokeDash=[5, 5],
            color='#adb5bd'
        ).encode(
            x='fiscal_year:O',
            y='constant_dollar_amount:Q',
            tooltip=[alt.Tooltip('fiscal_year', title='Year'), alt.Tooltip('constant_dollar_amount', title='Sector Avg', format='$,.2f')]
        )

        st.altair_chart(base + avg_line, use_container_width=True)
    else:
        st.info("No historical records found for this combination — prediction will rely on inferred patterns.")

    st.divider()

    # === PREDICTION SECTION ===
    if st.button("🔮 Generate Forecast"):
        if model_pipeline is not None:
            input_data = compute_features(df_history, fiscal_year, managing_agency_name, funding_agency_name, sector, is_refund)
            
            # Ensure all required columns for the pipeline are present (even if zero/defaulted)
            # This is a safety measure, though compute_features should handle it.
            required_cols = list(input_data.columns) # Assuming the pipeline was trained on these columns
            input_data = input_data[required_cols]

            log_pred = model_pipeline.predict(input_data)
            # Assuming the model predicts log(amount + 1), reverse the transformation
            prediction = np.expm1(log_pred)[0] 
            
            # Prevent negative predictions (though expm1 should prevent values < -1)
            prediction = max(0, prediction) 

            # Determine previous year's funding (if available)
            prev_funding = df_plot.loc[df_plot['fiscal_year'] == fiscal_year - 1, 'constant_dollar_amount']
            if not prev_funding.empty:
                prev_val = prev_funding.values[0]
                diff = prediction - prev_val
                pct_change = (diff / prev_val) * 100 if prev_val != 0 else 0
                trend_icon = "📈" if diff > 0 else "📉" if diff < 0 else "⏸️"
                trend_text = f"{trend_icon} {'Increase' if diff > 0 else 'Decrease' if diff < 0 else 'No Change'} of {abs(pct_change):.2f}% from {fiscal_year - 1}"
            else:
                trend_text = "No previous year data available for comparison."

            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #E6FFFA, #DFF6FF);
                padding: 2em;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            '>
                <h2 style='color:#004d61;margin-bottom:0;'>💵 ${prediction:,.2f}</h2>
                <p style='margin-top:0.5em;color:#007b83;font-size:17px;'>
                    Estimated U.S. Foreign Aid Allocation for <b>{sector}</b><br>
                    Managed by <b>{managing_agency_name}</b> in Fiscal Year <b>{fiscal_year}</b>.
                </p>
                <p style='font-size:15px;color:#006D77;'><i>{trend_text}</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model not loaded — unable to generate forecast.")

    # === FOOTER ===
    st.divider()
    st.markdown("""
    *Developed by **Ahjin Analytics*** """)
else:
    st.warning("Please upload `modeling data.csv` to enable the application inputs.")