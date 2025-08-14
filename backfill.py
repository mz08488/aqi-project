import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os
import numpy as np
from datetime import datetime, timedelta

end=datetime.now().date()
start=end-timedelta(days=2)


def fetch_past_data(latitude, longitude, city_name,start_date, end_date):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                   "sulphur_dioxide", "ozone", "dust", "uv_index", "us_aqi"],
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        # "forecast_days": 0,
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    for i, var in enumerate(params["hourly"]):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
    df = pd.DataFrame(data=hourly_data)
    df['city'] = city_name
    
    return df

def create_features(df):
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    df['aqi_lag_24'] = df['us_aqi'].shift(24)
    df['aqi_lag_48'] = df['us_aqi'].shift(48)
    
    df['aqi_rolling_24_mean'] = df['us_aqi'].rolling(window=24).mean()
    df['aqi_rolling_24_std'] = df['us_aqi'].rolling(window=24).std()
    
    df['aqi_change_rate'] = (df['us_aqi'] - df['aqi_lag_24']) / df['aqi_lag_24']
    
    # df.fillna(method='ffill', inplace=True)
    df.ffill(inplace=True)
    # Drop rows with NaN values after feature engineering   
    df = df.dropna()
    
    return df

def backfill_city_data(city_name, latitude, longitude):
    processed_file = f'data/{city_name}_historical_aq.csv'
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data file not found for {city_name}. Please run initial fetch script first.")
    
    # Load existing processed data
    existing_df = pd.read_csv(processed_file, parse_dates=['date'])
    
    # Fetch past 24h data and engineer features
    new_raw_df = fetch_past_data(latitude, longitude, city_name,start, end)
    new_processed_df = create_features(new_raw_df)
    print(f"Fetched {len(new_processed_df)} new rows for {city_name} from {start} to {end}.")
    
    # Avoid duplicates: keep only new rows with dates not in existing_df
    latest_existing_date = existing_df['date'].max()
    
    new_entries_df = new_processed_df[new_processed_df['date'] > latest_existing_date]
    
    if new_entries_df.empty:
        print(f"No new data to append for {city_name}.")
        return
    
    # Append new entries
    updated_df = pd.concat([existing_df, new_entries_df], ignore_index=True)
    
    # Optional: sort by date to keep order
    updated_df = updated_df.sort_values('date').reset_index(drop=True)
    
    # Save back
    updated_df.to_csv(processed_file, index=False)
    print(f"Backfilled and updated data saved for {city_name} with {len(new_entries_df)} new rows.")

if __name__ == "__main__":
    cities = {
        "Karachi": (24.8607, 67.0011),
        "Lahore": (31.5204, 74.3587),
        "Islamabad": (33.6844, 73.0479)
    }

    for city, (lat, lon) in cities.items():
        print(f"Backfilling data for {city}...")
        backfill_city_data(city, lat, lon)
