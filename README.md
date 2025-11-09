# Impact of Foreign Aid Cutoff on Kenya’s Socio-Economic Sectors

![Foreign Aid](https://unsplash.com/photos/red-and-white-x-logo-L8iPDE99z9c)

## Overview

Between **2010–2025**, the United States—primarily through **USAID**—has been a cornerstone of Kenya’s socio-economic development. However, the **2025 U.S. foreign aid freeze** caused major disruptions across health, education, and agriculture sectors, exposing the country’s vulnerability to donor dependency.

This repository presents a **comprehensive data-driven assessment** quantifying the impact of U.S. aid fluctuations on Kenya’s development and forecasting potential future outcomes using advanced regression and machine learning models.

## Project Objectives

1. **Quantify Aid Dependency**  
   Measure Kenya’s reliance on U.S. foreign aid at sector and agency levels.
2. **Assess Impact of Aid Cutoffs**  
   Analyze how funding reductions affect national performance indicators.
3. **Identify Vulnerable Sectors**  
   Determine which sectors are most exposed to future aid disruptions.
4. **Forecast Future Scenarios**  
   Build predictive models to anticipate sectoral and fiscal outcomes.

## Data Source

| Attribute | Description |
|------------|-------------|
| **Source** | [ForeignAssistance.gov](https://foreignassistance.gov/data#tab-data-download) |
| **Coverage** | 2000–2025 |
| **Records** | ~80,000 (filtered from 3M+ global entries) |
| **Features** | 56 attributes including sectors, agencies, aid types, fiscal years, and disbursements |

**Focus Country:** Kenya 🇰🇪  
**Currency:** Constant U.S. Dollars (Inflation-Adjusted)

## Data Preparation

Key preprocessing steps included:

- **Filtering** records for Kenya (2000–2025)
- **Dropping redundant fields** (IDs, codes, text duplicates)
- **Imputing missing dates** using fiscal year midpoints
- **Standardizing column names** to snake_case
- **Feature engineering**:
  - Rolling 3-year means & volatility indices
  - Annual growth rates & dependency ratios
  - Aid Diversity Index (Shannon Entropy)

**Final dataset:** ~79,000 clean records × 14 analytical columns

## Exploratory Data Analysis (EDA)

| Insight | Observation |
|----------|-------------|
| **Top Sectors** | Health leads with ~$22.2B, followed by Human Rights ($8.4B) and Security ($2.3B). |
| **Underfunded Areas** | Education, Environment, and Economic Development receive < $0.6B. |
| **Top Agencies** | USAID, Department of State, and HHS dominate aid delivery. |
| **Aid Volatility** | Highest in Health and Security — driven by global events. |
| **Aid Diversity** | Improved post-2010, reflecting broader but still concentrated aid patterns. |

## Modeling & Methodology

### Algorithms Implemented
- **Linear Models:** OLS, Weighted LS, Robust OLS, ElasticNet  
- **Ensemble Models:** Random Forest, XGBoost, CatBoost, LightGBM, Stacked Ensemble  
- **Hyperparameter Optimization:** RandomizedSearchCV, Optuna, TimeSeriesSplit CV

### Evaluation Metrics
- **R²** – Explained variance  
- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **Residual & Q-Q plots** for diagnostics  

### Model Performance Summary

| Model | MAE | RMSE | R² | Notes |
|--------|------|------|----|------|
| **XGBoost (Tuned, TimeSeriesSplit)** | 16,884.69 | 181,031.84 | **0.9922** | Best overall |
| **CatBoost (Tuned)** | 33,513.60 | 510,427.57 | 0.9380 | High accuracy |
| **ElasticNet** | — | — | 0.703 | Most interpretable |

**Feature Importance (Top 3):**
1. rolling_mean_3yr – Historical funding trends  
2. funding_growth_rate – Recent funding momentum  
3. lag_1 – Previous year’s disbursement  

## Key Insights

- **Aid volatility** strongly predicts future allocations (OLS β ≈ +0.77).  
- **Education, Environment, and Economy** remain systematically underfunded.  
- **USAID** is both the **largest** and **most volatile** donor.  
- **Predictive accuracy (R² ≈ 0.99)** allows early detection of potential fiscal shocks.  

## Strategic Implications

- **Policy Vulnerability:** Kenya’s overreliance on volatile agencies makes funding unpredictable.  
- **Governance Bias:** U.S. aid heavily favors short-term administrative and health programs.  
- **Predictive Modeling:** Machine learning provides actionable foresight for policy stability.  

## Model Limitations

- Relies on U.S. ForeignAssistance.gov data only.  
- Annual-level granularity masks short-term trends.  
- High correlation between sectors and agencies.  
- Models capture correlation, not direct causation.  

## Recommendations

1. **Diversify Donor Base:** Engage EU, AfDB, and Asian partners.  
2. **Mobilize Domestic Revenue:** Build national fiscal buffers.  
3. **Negotiate Multi-Year Frameworks:** Reduce volatility with long-term commitments.  
4. **Invest in Open Data:** Enable real-time tracking and analytics.  
5. **Institutionalize Predictive Modeling:** Use ML forecasts in Vision 2030 planning.  

## Tech Stack

| Category | Tools |
|-----------|-------|
| **Languages** | Python (3.10) |
| **Libraries** | pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, catboost, optuna |
| **Environment** | Jupyter Notebook |
| **Version Control** | Git & GitHub |
| **Visualization** | Matplotlib, Seaborn |

## Results Summary

- **R² ≈ 0.99** (XGBoost tuned)  
- **Most Volatile Agencies:** HHS, DoD, USAID  
- **Most Stable Agencies:** Peace Corps, Treasury, Commerce  
- **Top Vulnerable Sectors:** Health, Human Rights, Security  
- **Chronic Underfunding:** Education, Environment, Economy  

## Citation

> **Mwapea, N., Karuiki, P., Chesire, A., Ogolla, C., & Chol, E. (2025).**  
> *The Impact of Foreign Aid Cutoff on Kenya’s Socio-Economic Sectors.*  
> Data Analytics and Predictive Modeling Report, November 2025.

## License

This project is released under the [MIT License](LICENSE).  

