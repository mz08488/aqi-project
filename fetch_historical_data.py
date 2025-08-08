# import os
# import pandas as pd
# from dotenv import load_dotenv
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry
# from datetime import datetime, timedelta

# # Load environment variables
# load_dotenv()

# CITIES = {
#     "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
#     "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
#     "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
# }

# # # Fetch historical air quality data
# # def fetch_air_quality_data(latitude, longitude, past_days=82):
# #     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
# #     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
# #     openmeteo = openmeteo_requests.Client(session=retry_session)
# #     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
# #     params = {
# #         "latitude": latitude,
# #         "longitude": longitude,
# #         "hourly": ["pm10", "pm2_5"],
# #         "current": "us_aqi",
# #         "timezone": "auto",
# #         "past_days": past_days
# #     }
# #     responses = openmeteo.weather_api(url, params=params)
# #     return responses[0]

# # def compute_features(response, city):
# #     hourly = response.Hourly()
# #     hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
# #     hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
# #     df = pd.DataFrame({
# #         "date": pd.date_range(
# #             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
# #             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
# #             freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"),
# #         "pm10": hourly_pm10,
# #         "pm2_5": hourly_pm2_5,
# #         "city": city
# #     })
# #     df["us_aqi"] = response.Current().Variables(0).Value()
# #     df["hour"] = df["date"].dt.hour
# #     df["day"] = df["date"].dt.day
# #     df["month"] = df["date"].dt.month
# #     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
# #     df["target_aqi"] = df["us_aqi"].shift(-72)
# #     return df

# def fetch_air_quality_data(latitude, longitude, past_days=82):
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)
#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5", "us_aqi"],  # us_aqi added here
#         "timezone": "auto",
#         "past_days": past_days
#     }
#     responses = openmeteo.weather_api(url, params=params)
#     return responses[0]

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

#     df["hour"] = df["date"].dt.hour
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month
#     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
#     df["target_aqi"] = df["us_aqi"].shift(-72)  # 3-day future target

#     return df

# def save_features(df, city):
#     path = f"data/historical/aqi_features_{city.lower()}.csv"
#     os.makedirs("data/historical", exist_ok=True)
#     if os.path.exists(path):
#         old = pd.read_csv(path)
#         df = pd.concat([old, df]).drop_duplicates(subset=["date"])
#     df.to_csv(path, index=False)

# def run():
#     for city, coords in CITIES.items():
#         response = fetch_air_quality_data(coords["latitude"], coords["longitude"])
#         df = compute_features(response, city)
#         save_features(df, city)

# if __name__ == "__main__":
#     run()


import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

def fetch_past_data(latitude, longitude, city):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "dust", "methane", "ammonia"
        ],
        "past_days": 92
    }
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    data = {"date": dates}
    for i, name in enumerate(params['hourly']):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    df.to_csv(f"data/{city}_past_data.csv", index=False)

if __name__ == '__main__':
    cities = {"karachi": (24.8607, 67.0011), "lahore": (31.5497, 74.3436), "islamabad": (33.6844, 73.0479)}
    for city, coords in cities.items():
        fetch_past_data(coords[0], coords[1], city)