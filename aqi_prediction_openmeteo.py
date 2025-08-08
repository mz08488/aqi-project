# # import openmeteo_requests
# # import requests_cache
# # from retry_requests import retry
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta
# # import hopsworks
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# # import streamlit as st
# # import shap
# # import os

# # # Configuration
# # HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# # CITY = "Berlin"
# # LATITUDE = 52.52
# # LONGITUDE = 13.41

# # # 1. Fetch raw data from Open-Meteo API
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

# # # 2. Compute features and targets
# # def compute_features(response):
# #     # Process hourly data
# #     hourly = response.Hourly()
# #     hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
# #     hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()

# #     hourly_data = {
# #         "date": pd.date_range(
# #             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
# #             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
# #             freq=pd.Timedelta(seconds=hourly.Interval()),
# #             inclusive="left"
# #         ),
# #         "pm10": hourly_pm10,
# #         "pm2_5": hourly_pm2_5
# #     }
# #     df = pd.DataFrame(data=hourly_data)

# #     # Add current AQI
# #     current = response.Current()
# #     df["us_aqi"] = current.Variables(0).Value()  # Simplified: repeating current AQI for all rows

# #     # Extract time-based features
# #     df["hour"] = df["date"].dt.hour
# #     df["day"] = df["date"].dt.day
# #     df["month"] = df["date"].dt.month

# #     # Compute derived features
# #     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    
# #     # Target: AQI for next 3 days (72 hours ahead)
# #     df["target_aqi"] = df["us_aqi"].shift(-72)

# #     return df[["date", "hour", "day", "month", "pm10", "pm2_5", "aqi_change", "us_aqi", "target_aqi"]]

# # # 3. Store features in Hopsworks
# # def store_features(df):
# #     project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
# #     fs = project.get_feature_store()
    
# #     fg = fs.get_or_create_feature_group(
# #         name="aqi_features",
# #         version=1,
# #         primary_key=["date"],
# #         description="AQI prediction features for Berlin"
# #     )
# #     fg.insert(df)

# # # 4. Backfill historical data
# # def backfill_data(latitude, longitude, past_days=82):
# #     response = fetch_air_quality_data(latitude, longitude, past_days)
# #     df = compute_features(response)
# #     store_features(df)
# #     return df

# # # 5. Training pipeline
# # def train_model():
# #     project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
# #     fs = project.get_feature_store()
    
# #     fg = fs.get_feature_group(name="aqi_features", version=1)
# #     df = fg.read()
    
# #     # Prepare data
# #     X = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna()
# #     y = df["target_aqi"].dropna()
# #     X = X.loc[y.index]
    
# #     # Train Random Forest
# #     model = RandomForestRegressor(n_estimators=100, random_state=42)
# #     model.fit(X, y)
    
# #     # Evaluate
# #     y_pred = model.predict(X)
# #     rmse = np.sqrt(mean_squared_error(y, y_pred))
# #     mae = mean_absolute_error(y, y_pred)
# #     r2 = r2_score(y, y_pred)
    
# #     # Store model
# #     mr = project.get_model_registry()
# #     model_path = "aqi_model"
# #     model.save(model_path)
# #     mr_model = mr.create_model("aqi_model")
# #     mr_model.save(model_path)
    
# #     return model, {"rmse": rmse, "mae": mae, "r2": r2}

# # # 6. Feature importance with SHAP
# # def explain_model(model, X):
# #     explainer = shap.TreeExplainer(model)
# #     shap_values = explainer.shap_values(X)
# #     return shap_values

# # # 7. Web app with Streamlit
# # def run_web_app():
# #     st.title(f"{CITY} AQI Prediction")
    
# #     # Load model and features
# #     project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
# #     mr = project.get_model_registry()
# #     model = mr.get_model("aqi_model").load()
    
# #     fs = project.get_feature_store()
# #     fg = fs.get_feature_group(name="aqi_features", version=1)
# #     df = fg.read()
    
# #     # Get latest features
# #     latest_features = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].iloc[-1]
# #     prediction = model.predict([latest_features])[0]
    
# #     # Display prediction
# #     st.write(f"Predicted AQI for next 3 days: {prediction:.2f}")
    
# #     # Alert for hazardous AQI
# #     if prediction > 100:
# #         st.warning("Hazardous AQI level predicted!")
    
# #     # Display feature importance
# #     shap_values = explain_model(model, df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna())
# #     st.subheader("Feature Importance")
# #     shap.summary_plot(shap_values, df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]], plot_type="bar")

# # # Main execution
# # if __name__ == "__main__":
# #     # Run feature pipeline hourly
# #     response = fetch_air_quality_data(LATITUDE, LONGITUDE)
# #     features = compute_features(response)
# #     store_features(features)
    
# #     # Run training daily
# #     model, metrics = train_model()
# #     print(f"Model metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}")
    
# #     # Run web app
# #     run_web_app()






# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env file (if needed)
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import streamlit as st
# import shap
# import os

# # Configuration
# CITY = "Berlin"
# LATITUDE = 52.52
# LONGITUDE = 13.41
# FEATURE_FILE = "aqi_features.csv"
# MODEL_FILE = "aqi_model.pkl"

# # 1. Fetch raw data from Open-Meteo API
# def fetch_air_quality_data(latitude, longitude, past_days=82):
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)

#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5"],
#         "current": "us_aqi",
#         "timezone": "auto",
#         "past_days": past_days
#     }
#     responses = openmeteo.weather_api(url, params=params)
#     return responses[0]

# # 2. Compute features and targets
# def compute_features(response):
#     hourly = response.Hourly()
#     hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
#     hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()

#     hourly_data = {
#         "date": pd.date_range(
#             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#             freq=pd.Timedelta(seconds=hourly.Interval()),
#             inclusive="left"
#         ),
#         "pm10": hourly_pm10,
#         "pm2_5": hourly_pm2_5
#     }
#     df = pd.DataFrame(data=hourly_data)

#     # Add current AQI
#     current = response.Current()
#     df["us_aqi"] = current.Variables(0).Value()

#     # Extract time-based features
#     df["hour"] = df["date"].dt.hour
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month

#     # Compute derived features
#     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    
#     # Target: AQI for next 3 days (72 hours ahead)
#     df["target_aqi"] = df["us_aqi"].shift(-72)

#     return df[["date", "hour", "day", "month", "pm10", "pm2_5", "aqi_change", "us_aqi", "target_aqi"]]

# # 3. Store features locally in CSV
# def store_features(df):
#     if os.path.exists(FEATURE_FILE):
#         existing_df = pd.read_csv(FEATURE_FILE)
#         df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
#     df.to_csv(FEATURE_FILE, index=False)

# # 4. Backfill historical data
# def backfill_data(latitude, longitude, past_days=82):
#     response = fetch_air_quality_data(latitude, longitude, past_days)
#     df = compute_features(response)
#     store_features(df)
#     return df

# # 5. Training pipeline
# def train_model():
#     df = pd.read_csv(FEATURE_FILE)
    
#     # Prepare data
#     X = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna()
#     y = df["target_aqi"].dropna()
#     X = X.loc[y.index]
    
#     # Train Random Forest
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
    
#     # Evaluate
#     y_pred = model.predict(X)
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     mae = mean_absolute_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
    
#     # Save model locally
#     with open(MODEL_FILE, "wb") as f:
#         pickle.dump(model, f)
    
#     return model, {"rmse": rmse, "mae": mae, "r2": r2}

# # 6. Feature importance with SHAP
# def explain_model(model, X):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     return shap_values

# # 7. Web app with Streamlit
# def run_web_app():
#     st.title(f"{CITY} AQI Prediction")
    
#     # Load model and features
#     with open(MODEL_FILE, "rb") as f:
#         model = pickle.load(f)
    
#     df = pd.read_csv(FEATURE_FILE)
    
#     # Get latest features
#     latest_features = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].iloc[-1]
#     prediction = model.predict([latest_features])[0]
    
#     # Display prediction
#     st.write(f"Predicted AQI for next 3 days: {prediction:.2f}")
    
#     # Alert for hazardous AQI
#     if prediction > 100:
#         st.warning("Hazardous AQI level predicted!")
    
#     # Display feature importance
#     shap_values = explain_model(model, df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna())
#     st.subheader("Feature Importance")
#     shap.summary_plot(shap_values, df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]], plot_type="bar")

# # Main execution
# if __name__ == "__main__":
#     # Run feature pipeline
#     response = fetch_air_quality_data(LATITUDE, LONGITUDE)
#     features = compute_features(response)
#     store_features(features)
    
#     # Run training
#     model, metrics = train_model()
#     print(f"Model metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}")
    
#     # Run web app
#     run_web_app()

# --------------- working code----------------

# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env file (if needed)
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import streamlit as st
# import shap
# import matplotlib.pyplot as plt
# import os

# # Configuration
# CITY = "Berlin"
# LATITUDE = 52.52
# LONGITUDE = 13.41
# FEATURE_FILE = "aqi_features.csv"
# MODEL_FILE = "aqi_model.pkl"

# # 1. Fetch raw data from Open-Meteo API
# def fetch_air_quality_data(latitude, longitude, past_days=82):
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)

#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5"],
#         "current": "us_aqi",
#         "timezone": "auto",
#         "past_days": past_days
#     }
#     responses = openmeteo.weather_api(url, params=params)
#     return responses[0]

# # 2. Compute features and targets
# def compute_features(response):
#     hourly = response.Hourly()
#     hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
#     hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()

#     hourly_data = {
#         "date": pd.date_range(
#             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#             freq=pd.Timedelta(seconds=hourly.Interval()),
#             inclusive="left"
#         ),
#         "pm10": hourly_pm10,
#         "pm2_5": hourly_pm2_5
#     }
#     df = pd.DataFrame(data=hourly_data)

#     # Add current AQI
#     current = response.Current()
#     df["us_aqi"] = current.Variables(0).Value()

#     # Extract time-based features
#     df["hour"] = df["date"].dt.hour
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month

#     # Compute derived features
#     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    
#     # Target: AQI for next 3 days (72 hours ahead)
#     df["target_aqi"] = df["us_aqi"].shift(-72)

#     return df[["date", "hour", "day", "month", "pm10", "pm2_5", "aqi_change", "us_aqi", "target_aqi"]]

# # 3. Store features locally in CSV
# def store_features(df):
#     if os.path.exists(FEATURE_FILE):
#         existing_df = pd.read_csv(FEATURE_FILE)
#         df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
#     df.to_csv(FEATURE_FILE, index=False)

# # 4. Backfill historical data
# def backfill_data(latitude, longitude, past_days=82):
#     response = fetch_air_quality_data(latitude, longitude, past_days)
#     df = compute_features(response)
#     store_features(df)
#     return df

# # 5. Training pipeline
# def train_model():
#     df = pd.read_csv(FEATURE_FILE)
    
#     # Prepare data
#     X = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna()
#     y = df["target_aqi"].dropna()
#     X = X.loc[y.index]
    
#     # Train Random Forest
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
    
#     # Evaluate
#     y_pred = model.predict(X)
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     mae = mean_absolute_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
    
#     # Save model locally
#     with open(MODEL_FILE, "wb") as f:
#         pickle.dump(model, f)
    
#     return model, {"rmse": rmse, "mae": mae, "r2": r2}

# # 6. Feature importance with SHAP
# def explain_model(model, X):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     return shap_values

# # 7. Web app with Streamlit
# def run_web_app():
#     st.title(f"{CITY} AQI Prediction")
    
#     # Load model and features
#     try:
#         with open(MODEL_FILE, "rb") as f:
#             model = pickle.load(f)
#     except FileNotFoundError:
#         st.error("Model file not found. Please run the training pipeline first.")
#         return
    
#     try:
#         df = pd.read_csv(FEATURE_FILE)
#     except FileNotFoundError:
#         st.error("Feature file not found. Please run the feature pipeline first.")
#         return
    
#     # Get latest features
#     feature_columns = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]
#     latest_features = df[feature_columns].iloc[-1:]
#     prediction = model.predict(latest_features)[0]
    
#     # Display prediction
#     st.write(f"Predicted AQI for next 3 days: {prediction:.2f}")
    
#     # Alert for hazardous AQI
#     if prediction > 100:
#         st.warning("Hazardous AQI level predicted!")
    
#     # Display feature importance
#     st.subheader("Feature Importance")
#     shap_values = explain_model(model, df[feature_columns].dropna())
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, df[feature_columns].dropna(), plot_type="bar", show=False)
#     st.pyplot(fig)

# # Main execution
# if __name__ == "__main__":
#     # Run feature pipeline
#     response = fetch_air_quality_data(LATITUDE, LONGITUDE)
#     features = compute_features(response)
#     store_features(features)
    
#     # Run training
#     model, metrics = train_model()
#     print(f"Model metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}")
    
#     # Run web app
#     run_web_app()

# --------------- working code----------------

# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env file (if needed)
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import streamlit as st
# import shap
# import matplotlib.pyplot as plt
# import os

# # City configurations
# CITIES = {
#     "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
#     "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
#     "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
# }

# # 1. Fetch raw data from Open-Meteo API
# def fetch_air_quality_data(latitude, longitude, past_days=82):
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)

#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5"],
#         "current": "us_aqi",
#         "timezone": "auto",
#         "past_days": past_days
#     }
#     responses = openmeteo.weather_api(url, params=params)
#     return responses[0]

# # 2. Compute features and targets
# def compute_features(response):
#     hourly = response.Hourly()
#     hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
#     hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()

#     hourly_data = {
#         "date": pd.date_range(
#             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#             freq=pd.Timedelta(seconds=hourly.Interval()),
#             inclusive="left"
#         ),
#         "pm10": hourly_pm10,
#         "pm2_5": hourly_pm2_5
#     }
#     df = pd.DataFrame(data=hourly_data)

#     # Add current AQI
#     current = response.Current()
#     df["us_aqi"] = current.Variables(0).Value()

#     # Extract time-based features
#     df["hour"] = df["date"].dt.hour
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month

#     # Compute derived features
#     df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    
#     # Target: AQI for next 3 days (72 hours ahead)
#     df["target_aqi"] = df["us_aqi"].shift(-72)

#     return df[["date", "hour", "day", "month", "pm10", "pm2_5", "aqi_change", "us_aqi", "target_aqi"]]

# # 3. Store features locally in CSV
# def store_features(df, city):
#     feature_file = f"aqi_features_{city.lower()}.csv"
#     if os.path.exists(feature_file):
#         existing_df = pd.read_csv(feature_file)
#         df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
#     df.to_csv(feature_file, index=False)
#     return feature_file

# # 4. Backfill historical data
# def backfill_data(latitude, longitude, city, past_days=82):
#     response = fetch_air_quality_data(latitude, longitude, past_days)
#     df = compute_features(response)
#     feature_file = store_features(df, city)
#     return df, feature_file

# # 5. Training pipeline
# def train_model(feature_file):
#     df = pd.read_csv(feature_file)
    
#     # Prepare data
#     X = df[["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]].dropna()
#     y = df["target_aqi"].dropna()
#     X = X.loc[y.index]
    
#     # Train Random Forest
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
    
#     # Evaluate
#     y_pred = model.predict(X)
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     mae = mean_absolute_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
    
#     # Save model locally
#     model_file = f"aqi_model_{os.path.basename(feature_file).split('.')[0]}.pkl"
#     with open(model_file, "wb") as f:
#         pickle.dump(model, f)
    
#     return model, {"rmse": rmse, "mae": mae, "r2": r2}, model_file

# # 6. Feature importance with SHAP
# def explain_model(model, X):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     return shap_values

# # 7. Web app with Streamlit
# def run_web_app():
#     st.title("AQI Prediction for Pakistani Cities")
    
#     # City selection
#     city = st.selectbox("Select City", list(CITIES.keys()))
#     latitude = CITIES[city]["latitude"]
#     longitude = CITIES[city]["longitude"]
    
#     # Fetch and store data
#     response = fetch_air_quality_data(latitude, longitude)
#     features = compute_features(response)
#     feature_file = store_features(features, city)
    
#     # Train model
#     model, metrics, model_file = train_model(feature_file)
#     st.write(f"Model metrics for {city}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}")
    
#     # Load model and features
#     try:
#         with open(model_file, "rb") as f:
#             model = pickle.load(f)
#     except FileNotFoundError:
#         st.error("Model file not found. Please ensure the training pipeline has run.")
#         return
    
#     try:
#         df = pd.read_csv(feature_file)
#     except FileNotFoundError:
#         st.error("Feature file not found. Please run the feature pipeline first.")
#         return
    
#     # Get latest features
#     feature_columns = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]
#     latest_features = df[feature_columns].iloc[-1:]
#     prediction = model.predict(latest_features)[0]
    
#     # Display prediction
#     st.write(f"Predicted AQI for {city} (next 3 days): {prediction:.2f}")
    
#     # Alert for hazardous AQI
#     if prediction > 100:
#         st.warning(f"Hazardous AQI level predicted for {city}!")
    
#     # Display recent AQI data
#     st.subheader(f"Recent AQI Context for {city}")
#     if city == "Karachi":
#         st.write("On December 2, 2024, Karachi had an AQI of 96 (Moderate) with PM2.5 of 33 µg/m³. Historical data shows worsening trends: 38.5 µg/m³ in 2017, 33.7 µg/m³ in 2018, and 40.2 µg/m³ in 2019. Source: www.aqi.in, www.iqair.com")
#         st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/sindh/karachi)")
#     elif city == "Lahore":
#         st.write("On December 2, 2024, Lahore had an AQI of 88 (Moderate) with PM2.5 of 33 µg/m³. Historical improvements: 133.2 µg/m³ in 2017, 114.9 µg/m³ in 2018, 89.5 µg/m³ in 2019. Winter smog remains a challenge. Source: www.aqi.in, www.iqair.com")
#         st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/punjab/lahore)")
#     elif city == "Islamabad":
#         st.write("On December 2, 2024, Islamabad had an AQI of 97 (Moderate) with PM2.5 of 33 µg/m³. It ranks better than Lahore but worse than Karachi historically (2018: 239th globally). Source: www.aqi.in, www.iqair.com")
#         st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/islamabad)")
    
#     # Display feature importance
#     st.subheader(f"Feature Importance for {city}")
#     shap_values = explain_model(model, df[feature_columns].dropna())
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, df[feature_columns].dropna(), plot_type="bar", show=False)
#     st.pyplot(fig)

# # Main execution
# if __name__ == "__main__":
#     run_web_app()

from dotenv import load_dotenv
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import shap
import matplotlib.pyplot as plt
import os
import joblib

# Load environment variables (if needed for future extensions)
load_dotenv()

# City configurations
CITIES = {
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
    "Lahore": {"latitude": 31.5497, "longitude": 74.3436},
    "Islamabad": {"latitude": 33.6844, "longitude": 73.0479}
}

# 1. Fetch raw data from Open-Meteo API
def fetch_air_quality_data(latitude, longitude, past_days=82):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["pm10", "pm2_5"],
        "current": "us_aqi",
        "timezone": "auto",
        "past_days": past_days
    }
    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

# 2. Compute features and targets
def compute_features(response, city):
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly_pm10,
        "pm2_5": hourly_pm2_5,
        "city": city
    }
    df = pd.DataFrame(data=hourly_data)

    # Add current AQI
    current = response.Current()
    df["us_aqi"] = current.Variables(0).Value()

    # Extract time-based features
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Compute derived features
    df["aqi_change"] = df["us_aqi"].diff().fillna(0)
    
    # Target: AQI for next 3 days (72 hours ahead)
    df["target_aqi"] = df["us_aqi"].shift(-72)

    return df[["date", "hour", "day", "month", "pm10", "pm2_5", "aqi_change", "us_aqi", "target_aqi", "city"]]

# 3. Store features in CSV
def store_features(df, city):
    feature_file = f"aqi_features_{city.lower()}.csv"
    if os.path.exists(feature_file):
        existing_df = pd.read_csv(feature_file)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
    df.to_csv(feature_file, index=False)
    return feature_file

# 4. Backfill historical data
def backfill_data(latitude, longitude, city, past_days=82):
    response = fetch_air_quality_data(latitude, longitude, past_days)
    df = compute_features(response, city)
    feature_file = store_features(df, city)
    return df, feature_file

# 5. Training pipeline
def train_model(city):
    feature_file = f"aqi_features_{city.lower()}.csv"
    try:
        df = pd.read_csv(feature_file)
    except FileNotFoundError:
        raise Exception(f"Feature file for {city} not found. Run feature pipeline first.")
    
    # Prepare data
    feature_columns = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]
    X = df[feature_columns].dropna()
    y = df["target_aqi"].dropna()
    X = X.loc[y.index]
    
    # Train multiple models
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ridge": Ridge(alpha=1.0)
    }
    
    metrics = {}
    best_model = None
    best_r2 = -float("inf")
    best_model_name = ""
    
    # Train and evaluate models
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
    
    # Save best model
    model_file = f"aqi_model_{city.lower()}.pkl"
    joblib.dump(best_model, model_file)
    
    return best_model, metrics, best_model_name, model_file

# 6. Feature importance with SHAP
def explain_model(model, X, model_name):
    explainer = shap.TreeExplainer(model) if model_name == "random_forest" else shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    return shap_values

# 7. Web app with Streamlit
def run_web_app():
    st.title("AQI Prediction for Pakistani Cities")
    
    # City selection
    city = st.selectbox("Select City", list(CITIES.keys()))
    latitude = CITIES[city]["latitude"]
    longitude = CITIES[city]["longitude"]
    
    # Fetch and store latest data
    response = fetch_air_quality_data(latitude, longitude)
    features = compute_features(response, city)
    feature_file = store_features(features, city)
    
    # Train model
    try:
        model, metrics, model_name, model_file = train_model(city)
        st.write(f"Model metrics for {city}: RMSE={metrics[model_name]['rmse']:.2f}, MAE={metrics[model_name]['mae']:.2f}, R2={metrics[model_name]['r2']:.2f} (Model: {model_name.replace('_', ' ').title()})")
    except Exception as e:
        st.error(f"Error training model for {city}: {str(e)}")
        return
    
    # Load model
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        st.error(f"Model file for {city} not found.")
        return
    
    # Load features
    try:
        df = pd.read_csv(feature_file)
    except FileNotFoundError:
        st.error(f"Feature file for {city} not found.")
        return
    
    # Get latest features
    feature_columns = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]
    latest_features = df[feature_columns].iloc[-1:]
    prediction = model.predict(latest_features)[0]
    
    # Display prediction
    st.subheader(f"Predicted AQI for {city} (next 3 days)")
    st.write(f"{prediction:.2f} (Model: {model_name.replace('_', ' ').title()})")
    
    # Alert for hazardous AQI
    if prediction > 100:
        st.warning(f"Hazardous AQI level predicted for {city}!")
    
    # Display recent AQI data
    st.subheader(f"Recent AQI Context for {city}")
    if city == "Karachi":
        st.write("On December 2, 2024, Karachi had an AQI of 96 (Moderate) with PM2.5 of 33 µg/m³. Historical data shows worsening trends: 38.5 µg/m³ in 2017, 33.7 µg/m³ in 2018, 40.2 µg/m³ in 2019.")
        st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/sindh/karachi)")
    elif city == "Lahore":
        st.write("On December 2, 2024, Lahore had an AQI of 88 (Moderate) with PM2.5 of 33 µg/m³. Historical improvements: 133.2 µg/m³ in 2017, 114.9 µg/m³ in 2018, 89.5 µg/m³ in 2019. Winter smog remains a challenge.")
        st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/punjab/lahore)")
    elif city == "Islamabad":
        st.write("On December 2, 2024, Islamabad had an AQI of 97 (Moderate) with PM2.5 of 33 µg/m³. It ranks better than Lahore but worse than Karachi historically (2018: 239th globally).")
        st.markdown("[View more at AQI.in](https://www.aqi.in/us/dashboard/pakistan) | [View more at IQAir](https://www.iqair.com/pakistan/islamabad)")
    
    # Display feature importance
    st.subheader(f"Feature Importance for {city}")
    shap_values = explain_model(model, df[feature_columns].dropna(), model_name)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df[feature_columns].dropna(), plot_type="bar", show=False)
    st.pyplot(fig)
    
    # Basic EDA: AQI trend plot
    st.subheader(f"AQI Trend for {city}")
    fig, ax = plt.subplots()
    df.set_index("date")["us_aqi"].plot(ax=ax)
    ax.set_title(f"Historical AQI for {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    st.pyplot(fig)

# Main execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--feature-pipeline":
        for city, coords in CITIES.items():
            backfill_data(coords["latitude"], coords["longitude"], city)
    elif len(sys.argv) > 1 and sys.argv[1] == "--training-pipeline":
        for city in CITIES:
            train_model(city)
    else:
        run_web_app()