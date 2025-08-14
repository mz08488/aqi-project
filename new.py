import os
import pandas as pd
import joblib
import requests
from datetime import datetime

# Define cities and coordinates
CITIES = {
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
    "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
    "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
}

FEATURES = ['pm2_5', 'pm10', 'no2', 'co', 'so2', 'o3', 'day', 'month', 'year', 'dayofweek']

os.makedirs("data/forecast", exist_ok=True)

def compute_features(df, city):
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["city"] = city
    return df

def fetch_air_quality_data(lat, lon, past_days=1):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm2_5", "pm10", "carbon_monoxide", "ozone",
            "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "past_days": past_days,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["hourly"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"])
    df.rename(columns={
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3"
    }, inplace=True)

    return df

def forecast_and_predict():
    for city, coords in CITIES.items():
        df = fetch_air_quality_data(coords["latitude"], coords["longitude"], past_days=1)
        df = compute_features(df, city)
        df.dropna(inplace=True)

        forecast_file = f"data/forecast/forecast_features_{city.lower()}.csv"
        df.to_csv(forecast_file, index=False)

        model_path = f"models/{city.lower()}_random_forest.pkl"
        model = joblib.load(model_path)
        latest_features = df[FEATURES].iloc[-1:]
        prediction = model.predict(latest_features)[0]
        print(f"{city} → Predicted AQI (next 3 days): {prediction:.2f}")

def predict_live_forecast():
    lat, lon = 24.8607, 67.0011
    tz = "Asia/Karachi"
    forecast_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm2_5", "pm10", "carbon_monoxide", "ozone", "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "forecast_days": 1,
        "timezone": tz
    }
    response = requests.get(forecast_url, params=params)
    data = response.json()["hourly"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek

    df.rename(columns={
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3"
    }, inplace=True)

    forecast_X = df[FEATURES].dropna()

    if forecast_X.empty:
        print("⚠️ No complete forecast data available for prediction.")
        return

    model = joblib.load("models/karachi_random_forest.pkl")
    df["predicted_aqi"] = model.predict(forecast_X)

    output_path = "data/predicted_karachi_aqi_next24hr.csv"
    os.makedirs("data", exist_ok=True)
    df[["date", "predicted_aqi"]].to_csv(output_path, index=False)

    print(f"✅ Saved predicted AQI for next 24 hours to {output_path}")
    print(df[["date", "predicted_aqi"]].head())

if __name__ == "__main__":
    forecast_and_predict()
    predict_live_forecast()
