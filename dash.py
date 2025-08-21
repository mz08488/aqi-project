

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page config
st.set_page_config(page_title="AQI Predictor Pakistan", layout="wide")

# Title
st.title("Air Quality Index (AQI) Prediction for Pakistani Cities")

# Sidebar for city selection
city = st.sidebar.selectbox("Select City", ["Karachi", "Lahore", "Islamabad"])

# Load data
@st.cache_data
def load_data(city_name):
    try:
        df = pd.read_csv(f'data/{city_name}_aqi_forecast.csv', parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error(f"Data not found for {city_name}. Please run the data pipeline first.")
        return None

df = load_data(city)

if df is not None:
    # Filter for forecast period (last 3 days)
    forecast_start = df['date'].max() - pd.Timedelta(days=3)
    forecast_df = df[df['date'] >= forecast_start]
    
    # Current AQI
    current_aqi = forecast_df.iloc[-1]['random_forest_pred']
    st.metric(f"Current Predicted AQI for {city}", f"{current_aqi:.1f}")
    
    # AQI Interpretation
    aqi_level = "Good" if current_aqi < 50 else \
                "Moderate" if current_aqi < 100 else \
                "Unhealthy for Sensitive Groups" if current_aqi < 150 else \
                "Unhealthy" if current_aqi < 200 else \
                "Very Unhealthy" if current_aqi < 300 else "Hazardous"
    
    st.write(f"**AQI Level:** {aqi_level}")
    
    # Plot predictions
    st.subheader("AQI Predictions (Last 3 Days)")
    fig = px.line(forecast_df, x='date', y=['random_forest_pred', 'ridge_pred', 'svr_pred'],
                  labels={'value': 'AQI', 'date': 'Date'},
                  title=f'AQI Predictions for {city}')
    st.plotly_chart(fig, use_container_width=True)
 # Model comparison
st.subheader("Model Comparison")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", f"{forecast_df['random_forest_pred'].mean():.1f}")
with col2:
    st.metric("Ridge Regression", f"{forecast_df['ridge_pred'].mean():.1f}")
with col3:
    st.metric("Support Vector", f"{forecast_df['svr_pred'].mean():.1f}")

# Raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(forecast_df)

# 🔑 SHAP Explainability should be INSIDE the same block as forecast_df
st.subheader("Feature Importance (SHAP)")

model_choice = st.selectbox(
    "Select Model for SHAP Analysis",
    ["random_forest", "ridge"]  # removed "svr"
)

shap_summary_path = f"shap_plots/{city}_{model_choice}_summary.png"
shap_bar_path = f"shap_plots/{city}_{model_choice}_bar.png"

try:
    st.write("**SHAP Summary Plot** (impact of features across predictions)")
    st.image(shap_summary_path, use_container_width=True)

    st.write("**SHAP Bar Plot** (average feature importance)")
    st.image(shap_bar_path, use_container_width=True)

except FileNotFoundError:
    st.warning(f"No SHAP plots found for {city} ({model_choice}). Please run the pipeline first.")

# # -------------------------
# # Exploratory Data Analysis
# # -------------------------
# st.subheader("Exploratory Data Analysis (EDA)")

# # Basic statistics
# st.write("### Descriptive Statistics")
# st.dataframe(forecast_df[['random_forest_pred', 'ridge_pred', 'svr_pred']].describe())

# # AQI distribution plots
# st.write("### AQI Distributions")
# fig_dist = px.histogram(
#     forecast_df, 
#     x=["random_forest_pred", "ridge_pred", "svr_pred"],
#     marginal="box", 
#     barmode="overlay",
#     opacity=0.6,
#     nbins=30,
#     labels={"value": "AQI"}
# )
# st.plotly_chart(fig_dist, use_container_width=True)

# # Trend comparison
# st.write("### AQI Trends Over Time")
# fig_trend = px.line(
#     forecast_df, 
#     x="date", 
#     y=["random_forest_pred", "ridge_pred", "svr_pred"],
#     title=f"Model Trends for {city}"
# )
# st.plotly_chart(fig_trend, use_container_width=True)

# # Correlation heatmap
# st.write("### Correlation Between Models")
# corr = forecast_df[['random_forest_pred', 'ridge_pred', 'svr_pred']].corr()
# fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Model Correlation Heatmap")
# st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Exploratory Data Analysis
# -------------------------
st.subheader("Exploratory Data Analysis (EDA)")

# Load historical data (with features)
@st.cache_data
def load_historical(city_name):
    try:
        df_hist = pd.read_csv(f"data/{city_name}_historical_aq.csv", parse_dates=["date"])
        return df_hist
    except FileNotFoundError:
        st.warning(f"No historical data found for {city_name}. Please run the pipeline first.")
        return None

hist_df = load_historical(city)

if hist_df is not None:
    # Show descriptive statistics
    st.write("### Descriptive Statistics of Historical Features")
    st.dataframe(hist_df.describe())

    # AQI distribution
    st.write("### AQI Distribution (Historical)")
    fig_hist_dist = px.histogram(
        hist_df, x="us_aqi", nbins=40, 
        marginal="box", opacity=0.7,
        title=f"AQI Distribution for {city}"
    )
    st.plotly_chart(fig_hist_dist, use_container_width=True)

    # AQI trend over time
    st.write("### AQI Trend Over Time (Historical)")
    fig_hist_trend = px.line(
        hist_df, x="date", y="us_aqi",
        title=f"AQI Trend for {city}"
    )
    st.plotly_chart(fig_hist_trend, use_container_width=True)

    # Correlation heatmap of features
    st.write("### Feature Correlation with AQI")
    corr = hist_df[["us_aqi","pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
                    "sulphur_dioxide","ozone","dust","uv_index"]].corr()

    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu",
        title=f"Correlation Heatmap ({city})"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter plots: feature vs AQI
    st.write("### Key Features vs AQI")
    feature_choice = st.selectbox(
        "Select feature to compare with AQI:",
        ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
         "sulphur_dioxide","ozone","dust","uv_index"]
    )
    fig_scatter = px.scatter(
        hist_df, x=feature_choice, y="us_aqi",
        opacity=0.6, trendline="ols",
        title=f"{feature_choice} vs AQI ({city})"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
