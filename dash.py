# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np

# # Page config
# st.set_page_config(page_title="AQI Predictor Pakistan", layout="wide")

# # Title
# st.title("Air Quality Index (AQI) Prediction for Pakistani Cities")

# # Sidebar for city selection
# city = st.sidebar.selectbox("Select City", ["Karachi", "Lahore", "Islamabad"])

# # Load data
# @st.cache_data
# def load_data(city_name):
#     try:
#         df = pd.read_csv(f'data/{city_name}_predictions.csv', parse_dates=['date'])
#         return df
#     except FileNotFoundError:
#         st.error(f"Data not found for {city_name}. Please run the data pipeline first.")
#         return None

# df = load_data(city)

# if df is not None:
#     # Filter for forecast period (last 3 days)
#     forecast_start = df['date'].max() - pd.Timedelta(days=3)
#     forecast_df = df[df['date'] >= forecast_start]
    
#     # Current AQI
#     current_aqi = forecast_df.iloc[-1]['random_forest_pred']
#     st.metric(f"Current Predicted AQI for {city}", f"{current_aqi:.1f}")
    
#     # AQI Interpretation
#     aqi_level = "Good" if current_aqi < 50 else \
#                 "Moderate" if current_aqi < 100 else \
#                 "Unhealthy for Sensitive Groups" if current_aqi < 150 else \
#                 "Unhealthy" if current_aqi < 200 else \
#                 "Very Unhealthy" if current_aqi < 300 else "Hazardous"
    
#     st.write(f"**AQI Level:** {aqi_level}")
    
#     # Plot predictions
#     st.subheader("AQI Predictions (Last 3 Days)")
#     fig = px.line(forecast_df, x='date', y=['random_forest_pred', 'ridge_pred', 'neural_network_pred'],
#                   labels={'value': 'AQI', 'date': 'Date'},
#                   title=f'AQI Predictions for {city}')
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Model comparison
#     st.subheader("Model Comparison")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Random Forest", f"{forecast_df['random_forest_pred'].mean():.1f}")
#     with col2:
#         st.metric("Ridge Regression", f"{forecast_df['ridge_pred'].mean():.1f}")
#     with col3:
#         st.metric("Neural Network", f"{forecast_df['neural_network_pred'].mean():.1f}")
    
#     # Raw data
#     if st.checkbox("Show raw data"):
#         st.subheader("Raw Data")
#         st.dataframe(forecast_df)
# else:
#     st.warning("No data available. Please run the data pipeline first.")



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
        df = pd.read_csv(f'data/{city_name}_predictions.csv', parse_dates=['date'])
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
else:
    st.warning("No data available. Please run the data pipeline first.")