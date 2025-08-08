# import openmeteo_requests
# import pandas as pd
# import requests_cache
# from retry_requests import retry
# import os

# def fetch_air_quality_data(latitude, longitude, city_name):
#     # Setup the Open-Meteo API client
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)

#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
#                    "sulphur_dioxide", "ozone", "dust", "uv_index"],
#         "past_days": 92,
#         "forecast_days": 3,
#     }
    
#     responses = openmeteo.weather_api(url, params=params)
#     response = responses[0]
    
#     # Process hourly data
#     hourly = response.Hourly()
#     hourly_data = {
#         "date": pd.date_range(
#             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#             freq=pd.Timedelta(seconds=hourly.Interval()),
#             inclusive="left"
#         )
#     }
    
#     # Add all variables
#     for i, var in enumerate(params["hourly"]):
#         hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
#     df = pd.DataFrame(data=hourly_data)
#     df['city'] = city_name
    
#     # Save to CSV
#     os.makedirs('data', exist_ok=True)
#     df.to_csv(f'data/{city_name}_air_quality.csv', index=False)
#     return df

# # Cities in Pakistan with coordinates
# cities = {
#     "Karachi": (24.8607, 67.0011),
#     "Lahore": (31.5204, 74.3587),
#     "Islamabad": (33.6844, 73.0479)
# }

# if __name__ == "__main__":
#     for city, (lat, lon) in cities.items():
#         print(f"Fetching data for {city}...")
#         fetch_air_quality_data(lat, lon, city)
        

# final try --------------------]

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

def fetch_air_quality_data(latitude, longitude, city_name):
    # Setup the Open-Meteo API client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                  "sulphur_dioxide", "ozone", "dust", "uv_index"],
        "past_days": 92,
        "forecast_days": 3,
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    # Add all variables
    for i, var in enumerate(params["hourly"]):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
    df = pd.DataFrame(data=hourly_data)
    df['city'] = city_name
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv(f'data/{city_name}_air_quality.csv', index=False)
    return df

# Cities in Pakistan with coordinates
cities = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479)
}

if __name__ == "__main__":
    for city, (lat, lon) in cities.items():
        print(f"Fetching data for {city}...")
        fetch_air_quality_data(lat, lon, city)