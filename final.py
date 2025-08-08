# import os
# import pandas as pd
# import numpy as np
# import requests
# import openmeteo_requests
# from retry_requests import retry
# import requests_cache
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# import plotly.express as px
# from datetime import datetime, timedelta
# from math import sqrt
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# CITIES = {
#     "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
#     "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
#     "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
# }
# FEATURES = ["pm10", "pm2_5", "hour", "day", "month", "dayofweek", "aqi_change"]
# TARGET = "target_aqi"

# # Fetch air quality data
# def fetch_air_quality_data(latitude, longitude, past_days=82):
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)
#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5", "us_aqi"],
#         "timezone": "auto",
#         "past_days": past_days
#     }
#     responses = openmeteo.weather_api(url, params=params)
#     return responses[0]

# # Compute features
# def compute_features(response, city):
#     hourly = response.Hourly()
#     pm10 = hourly.Variables(0).ValuesAsNumpy()
#     pm2_5 = hourly.Variables(1).ValuesAsNumpy()
#     us_aqi = hourly.Variables(2).ValuesAsNumpy()

#     time_index = pd.date_range(
#         start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#         end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#         freq=pd.Timedelta(seconds=hourly.Interval()),
#         inclusive="left"
#     )

#     df = pd.DataFrame({
#         "date": time_index,
#         "pm10": pm10,
#         "pm2_5": pm2_5,
#         "us_aqi": us_aqi,
#         "city": city
#     })

#     # Time-based and derived features
#     df["hour"] = df["date"].dt.hour
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month
#     df["dayofweek"] = df["date"].dt.dayofweek
#     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
#     df["target_aqi"] = df["us_aqi"].shift(-72)  # 3-day future target

#     return df

# # Save features to CSV
# def save_features(df, city):
#     path = f"data/historical/aqi_features_{city.lower()}.csv"
#     os.makedirs("data/historical", exist_ok=True)
#     if os.path.exists(path):
#         old = pd.read_csv(path)
#         df = pd.concat([old, df]).drop_duplicates(subset=["date"])
#     df.to_csv(path, index=False)

# # Perform EDA
# def perform_eda(df, city):
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(data=df, x="date", y="us_aqi", hue="city")
#     plt.title(f"AQI Trend for {city}")
#     plt.savefig(f"eda_aqi_trend_{city.lower()}.png")
#     plt.close()

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm")
#     plt.title(f"Feature Correlation for {city}")
#     plt.savefig(f"eda_correlation_{city.lower()}.png")
#     plt.close()

#     # Create a Chart.js line chart for AQI trend

#     {
#         "type": "line",
#         "data": {
#             "labels": [],
#             "datasets": [{
#                 "label": "AQI Trend",
#                 "data": [],
#                 "borderColor": "#636EFA",
#                 "backgroundColor": "rgba(99, 110, 250, 0.2)",
#                 "fill": True
#             }]
#         },
#         "options": {
#             "scales": {
#                 "x": {
#                     "title": {
#                         "display": True,
#                         "text": "Date"
#                     }
#                 },
#                 "y": {
#                     "title": {
#                         "display": True,
#                         "text": "AQI"
#                     }
#                 }
#             },
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": "AQI Trend for " + city
#                 }
#             }
#         }
#     }
    

# # Train and evaluate models
# def train_and_evaluate(city):
#     path = f"data/historical/aqi_features_{city.lower()}.csv"
#     df = pd.read_csv(path)
#     df.dropna(subset=[TARGET], inplace=True)

#     X = df[FEATURES]
#     y = df[TARGET]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Random Forest
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     rf_pred = rf_model.predict(X_test)
#     rf_metrics = {
#         "MAE": mean_absolute_error(y_test, rf_pred),
#         "RMSE": sqrt(mean_squared_error(y_test, rf_pred)),
#         "R2": r2_score(y_test, rf_pred)
#     }
#     joblib.dump(rf_model, f"models/{city.lower()}_random_forest.pkl")

#     # Ridge Regression
#     ridge_model = Ridge(alpha=1.0)
#     ridge_model.fit(X_train, y_train)
#     ridge_pred = ridge_model.predict(X_test)
#     ridge_metrics = {
#         "MAE": mean_absolute_error(y_test, ridge_pred),
#         "RMSE": sqrt(mean_squared_error(y_test, ridge_pred)),
#         "R2": r2_score(y_test, ridge_pred)
#     }
#     joblib.dump(ridge_model, f"models/{city.lower()}_ridge.pkl")

#     # SHAP for Random Forest
#     explainer = shap.TreeExplainer(rf_model)
#     shap_values = explainer.shap_values(X_test)
#     shap.summary_plot(shap_values, X_test, show=False)
#     plt.savefig(f"shap_summary_{city.lower()}.png")
#     plt.close()

#     print(f"{city} Metrics - RF: {rf_metrics}, Ridge: {ridge_metrics}")

# # Forecast and predict
# def forecast_and_predict(city):
#     coords = CITIES[city]
#     response = fetch_air_quality_data(coords["latitude"], coords["longitude"], past_days=1)
#     df = compute_features(response, city)
#     df.dropna(inplace=True)

#     path = f"data/forecast/forecast_features_{city.lower()}.csv"
#     os.makedirs("data/forecast", exist_ok=True)
#     df.to_csv(path, index=False)

#     model = joblib.load(f"models/{city.lower()}_random_forest.pkl")
#     prediction = model.predict(df[FEATURES].iloc[-1:])[0]

#     # AQI Alerts
#     alert = None
#     if prediction > 150:
#         alert = f"⚠️ Hazardous AQI predicted for {city}: {prediction:.2f}"
#     elif prediction > 100:
#         alert = f"⚠️ Unhealthy AQI predicted for {city}: {prediction:.2f}"
    
#     return prediction, df, alert

# # Streamlit Dashboard
# def run_dashboard():
#     st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")
#     st.title("🌍 AQI Forecast Dashboard")

#     city = st.sidebar.selectbox("Select a City", list(CITIES.keys()))
#     prediction, df, alert = forecast_and_predict(city)

#     if alert:
#         st.warning(alert)

#     st.subheader(f"🧭 AQI Forecast for {city} (Next 72 Hours)")
#     st.metric(label="Predicted AQI", value=f"{prediction:.1f}")

#     fig = px.line(df, x="date", y="us_aqi", title=f"AQI Trend - {city}", markers=True, color_discrete_sequence=["#636EFA"])
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("📋 Forecast Data")
#     st.dataframe(df[["date", "us_aqi"]], use_container_width=True)

#     csv_download = df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "⬇️ Download Data as CSV",
#         data=csv_download,
#         file_name=f"aqi_forecast_{city.lower()}.csv",
#         mime="text/csv"
#     )

# # Main execution
# if __name__ == "__main__":
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("data", exist_ok=True)
    
#     # Command-line argument handling for CI/CD
#     import sys
#     mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
#     if mode in ["all", "feature-pipeline"]:
#         for city in CITIES:
#             response = fetch_air_quality_data(CITIES[city]["latitude"], CITIES[city]["longitude"])
#             df = compute_features(response, city)
#             perform_eda(df, city)
#             save_features(df, city)
    
#     if mode in ["all", "training-pipeline"]:
#         for city in CITIES:
#             train_and_evaluate(city)
    
#     if mode == "all":
#         run_dashboard()

import os
import pandas as pd
import numpy as np
import requests
import openmeteo_requests
from retry_requests import retry
import requests_cache
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from math import sqrt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CITIES = {
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
    "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
    "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
}
FEATURES = ["pm10", "pm2_5", "hour", "day", "month", "dayofweek", "aqi_change"]
TARGET = "target_aqi"

# Fetch air quality data
def fetch_air_quality_data(latitude, longitude, past_days=82):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["pm10", "pm2_5", "us_aqi"],
        "timezone": "auto",
        "past_days": past_days
    }
    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

# Compute features
def compute_features(response, city):
    hourly = response.Hourly()
    pm10 = hourly.Variables(0).ValuesAsNumpy()
    pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    us_aqi = hourly.Variables(2).ValuesAsNumpy()

    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    df = pd.DataFrame({
        "date": time_index,
        "pm10": pm10,
        "pm2_5": pm2_5,
        "us_aqi": us_aqi,
        "city": city
    })

    # Time-based and derived features
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    df["target_aqi"] = df["us_aqi"].shift(-72)  # 3-day future target

    return df

# Save features to CSV
def save_features(df, city):
    path = f"data/historical/aqi_features_{city.lower()}.csv"
    os.makedirs("data/historical", exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df]).drop_duplicates(subset=["date"])
    df.to_csv(path, index=False)

# Perform EDA
def perform_eda(df, city):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="date", y="us_aqi", hue="city")
    plt.title(f"AQI Trend for {city}")
    plt.savefig(f"eda_aqi_trend_{city.lower()}.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm")
    plt.title(f"Feature Correlation for {city}")
    plt.savefig(f"eda_correlation_{city.lower()}.png")
    plt.close()

    # Create a Chart.js line chart for AQI trend
    
    {
        "type": "line",
        "data": {
            "labels": [],
            "datasets": [{
                "label": "AQI Trend",
                "data": [],
                "borderColor": "#636EFA",
                "backgroundColor": "rgba(99, 110, 250, 0.2)",
                "fill": True
            }]
        },
        "options": {
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Date"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "AQI"
                    }
                }
            },
            "plugins": {
                "title": {
                    "display": True,
                    "text": "AQI Trend for " + city
                }
            }
        }
    }
    

# Train and evaluate models
def train_and_evaluate(city):
    path = f"data/historical/aqi_features_{city.lower()}.csv"
    df = pd.read_csv(path)
    df.dropna(subset=[TARGET], inplace=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        "MAE": mean_absolute_error(y_test, rf_pred),
        "RMSE": sqrt(mean_squared_error(y_test, rf_pred)),
        "R2": r2_score(y_test, rf_pred)
    }
    joblib.dump(rf_model, f"models/{city.lower()}_random_forest.pkl")

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_metrics = {
        "MAE": mean_absolute_error(y_test, ridge_pred),
        "RMSE": sqrt(mean_squared_error(y_test, ridge_pred)),
        "R2": r2_score(y_test, ridge_pred)
    }
    joblib.dump(ridge_model, f"models/{city.lower()}_ridge.pkl")

    # SHAP for Random Forest
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"shap_summary_{city.lower()}.png")
    plt.close()

    print(f"{city} Metrics - RF: {rf_metrics}, Ridge: {ridge_metrics}")

# Forecast and predict
def forecast_and_predict(city):
    coords = CITIES[city]
    response = fetch_air_quality_data(coords["latitude"], coords["longitude"], past_days=1)
    df = compute_features(response, city)
    df.dropna(inplace=True)

    path = f"data/forecast/forecast_features_{city.lower()}.csv"
    os.makedirs("data/forecast", exist_ok=True)
    df.to_csv(path, index=False)

    model = joblib.load(f"models/{city.lower()}_random_forest.pkl")
    prediction = model.predict(df[FEATURES].iloc[-1:])[0]

    # AQI Alerts
    alert = None
    if prediction > 150:
        alert = f"⚠️ Hazardous AQI predicted for {city}: {prediction:.2f}"
    elif prediction > 100:
        alert = f"⚠️ Unhealthy AQI predicted for {city}: {prediction:.2f}"
    
    return prediction, df, alert

# Streamlit Dashboard
def run_dashboard():
    st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")
    st.title("🌍 AQI Forecast Dashboard")

    city = st.sidebar.selectbox("Select a City", list(CITIES.keys()))
    prediction, df, alert = forecast_and_predict(city)

    if alert:
        st.warning(alert)

    st.subheader(f"🧭 AQI Forecast for {city} (Next 72 Hours)")
    st.metric(label="Predicted AQI", value=f"{prediction:.1f}")

    fig = px.line(df, x="date", y="us_aqi", title=f"AQI Trend - {city}", markers=True, color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast Data")
    st.dataframe(df[["date", "us_aqi"]], use_container_width=True)

    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data as CSV",
        data=csv_download,
        file_name=f"aqi_forecast_{city.lower()}.csv",
        mime="text/csv"
    )

# Main execution
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Command-line argument handling for CI/CD
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if mode in ["all", "feature-pipeline"]:
        for city in CITIES:
            response = fetch_air_quality_data(CITIES[city]["latitude"], CITIES[city]["longitude"])
            df = compute_features(response, city)
            perform_eda(df, city)
            save_features(df, city)
    
    if mode in ["all", "training-pipeline"]:
        for city in CITIES:
            train_and_evaluate(city)
    
    if mode == "all":
        run_dashboard()