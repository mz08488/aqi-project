# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from feature_engineering import prepare_data

# def load_models(city_name):
#     models = {}
    
#     # Load Random Forest
#     models['random_forest'] = joblib.load(f'models/{city_name}_random_forest.joblib')
    
#     # Load Ridge Regression
#     models['ridge'] = joblib.load(f'models/{city_name}_ridge.joblib')
    
#     # Load Neural Network
#     models['neural_network'] = tf.keras.models.load_model(f'models/{city_name}_neural_network.h5')
    
#     # Load Scaler
#     scaler = joblib.load(f'models/{city_name}_scaler.joblib')
    
#     return models, scaler

# def make_predictions(city_name):
#     # Prepare data (this will include forecast data)
#     X, y, df = prepare_data(city_name)
    
#     # Load models
#     models, scaler = load_models(city_name)
    
#     # Make predictions
#     predictions = {}
#     for name, model in models.items():
#         if name == 'neural_network':
#             pred = model.predict(X).flatten()
#         else:
#             pred = model.predict(X)
#         predictions[name] = pred
    
#     # Create DataFrame with predictions
#     results = df[['date', 'city']].copy()
#     for name, pred in predictions.items():
#         results[f'{name}_pred'] = pred
    
#     # Save predictions
#     results.to_csv(f'data/{city_name}_predictions.csv', index=False)
    
#     return results

# if __name__ == "__main__":
#     for city in ["Karachi", "Lahore", "Islamabad"]:
#         print(f"Making predictions for {city}...")
#         predictions = make_predictions(city)
#         print(predictions.tail())



# --------------- final try ----------------



# -------------------Asal Try----------------

# import os
# import pandas as pd
# import numpy as np
# import joblib
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry
# import logging
# import requests_cache
# # Logging setup for CI/CD
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("forecast.log"),
#         logging.StreamHandler()
#     ]
# )

# def setup_openmeteo_client():
#     """Set up Open-Meteo client with cache and retries."""
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     return openmeteo_requests.Client(session=retry_session)

# def fetch_forecast_data(latitude, longitude, city_name):
#     """Fetch 72-hour forecast data from Open-Meteo."""
#     logging.info(f"Fetching forecast data for {city_name}...")
#     openmeteo = setup_openmeteo_client()
#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
#                    "sulphur_dioxide", "ozone", "dust", "uv_index", "us_aqi"],
#         "forecast_days": 3,
#     }
    
#     try:
#         responses = openmeteo.weather_api(url, params=params)
#         response = responses[0]
#     except Exception as e:
#         logging.error(f"Error fetching forecast data for {city_name}: {str(e)}")
#         raise
    
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
    
#     for i, var in enumerate(params["hourly"]):
#         hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
#     df = pd.DataFrame(data=hourly_data)
#     df['city'] = city_name
#     return df

# def create_features(df, historical_df):
#     """Create forecast features using historical data."""
#     # Add historical data for lags and moving averages
#     last_48_hours = historical_df.tail(48)[['date', 'us_aqi']].copy()
#     forecast_df = df[['date', 'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
#                      'sulphur_dioxide', 'ozone', 'dust', 'uv_index', 'city']].copy()
    
#     # Combine historical and forecast data for continuity
#     combined_df = pd.concat([last_48_hours, forecast_df], ignore_index=True)
    
#     # Feature creation
#     combined_df['hour'] = combined_df['date'].dt.hour
#     combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
#     combined_df['month'] = combined_df['date'].dt.month
    
#     combined_df['aqi_lag_24'] = combined_df['us_aqi'].shift(24)
#     combined_df['aqi_lag_48'] = combined_df['us_aqi'].shift(48)
#     combined_df['aqi_rolling_24_mean'] = combined_df['us_aqi'].rolling(window=24).mean()
#     combined_df['aqi_rolling_24_std'] = combined_df['us_aqi'].rolling(window=24).std()
#     combined_df['aqi_change_rate'] = (combined_df['us_aqi'] - combined_df['aqi_lag_24']) / combined_df['aqi_lag_24']
    
#     # Fill missing values
#     combined_df = combined_df.ffill().bfill()
    
#     # Select only forecast data (last 72 hours)
#     forecast_df = combined_df.tail(72).copy()
#     return forecast_df

# def load_models(city_name):
#     """Load models and scaler for the given city."""
#     models = {}
#     model_dir = "models"
#     try:
#         # Random Forest
#         rf_path = os.path.join(model_dir, f"{city_name}_random_forest.joblib")
#         if os.path.exists(rf_path):
#             models['random_forest'] = joblib.load(rf_path)
#         else:
#             raise FileNotFoundError(f"Random Forest model for {city_name} not found: {rf_path}")

#         # Ridge
#         ridge_path = os.path.join(model_dir, f"{city_name}_ridge.joblib")
#         if os.path.exists(ridge_path):
#             ridge, ridge_scaler = joblib.load(ridge_path)
#             models['ridge'] = {"model": ridge, "scaler": ridge_scaler}
#         else:
#             raise FileNotFoundError(f"Ridge model for {city_name} not found: {ridge_path}")

#         # SVR
#         svr_path = os.path.join(model_dir, f"{city_name}_svr.joblib")
#         if os.path.exists(svr_path):
#             svr, svr_scaler = joblib.load(svr_path)
#             models['svr'] = {"model": svr, "scaler": svr_scaler}
#         else:
#             raise FileNotFoundError(f"SVR model for {city_name} not found: {svr_path}")

#     except Exception as e:
#         logging.error(f"Error loading models for {city_name}: {str(e)}")
#         raise
#     return models

# def make_predictions(city_name, forecast_df, models):
#     """Generate 72-hour predictions using loaded models."""
#     logging.info(f"Generating predictions for {city_name}...")
    
#     # Prepare data for prediction
#     feature_columns = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
#                       'sulphur_dioxide', 'ozone', 'dust', 'uv_index', 
#                       'hour', 'day_of_week', 'month', 
#                       'aqi_lag_24', 'aqi_lag_48', 'aqi_rolling_24_mean', 'aqi_rolling_24_std', 
#                       'aqi_change_rate']
    
#     X = forecast_df[feature_columns].select_dtypes(include=[np.number])
    
#     # Ensure no NaN values
#     if X.isna().any().any():
#         logging.warning(f"NaN values found in forecast data for {city_name}, filling...")
#         X = X.ffill().bfill()
    
#     predictions = {}
#     for model_name, model in models.items():
#         try:
#             if model_name in ['ridge', 'svr']:
#                 X_scaled = model['scaler'].transform(X)
#                 pred = model['model'].predict(X_scaled)
#             else:
#                 pred = model.predict(X)
#             predictions[model_name] = pred
#         except Exception as e:
#             logging.error(f"Error predicting with {model_name} for {city_name}: {str(e)}")
#             raise
    
#     # Build result DataFrame
#     results = forecast_df[['date', 'city']].copy()
#     for model_name, pred in predictions.items():
#         results[f'{model_name}_pred'] = pred
    
#     return results

# def save_predictions(city_name, predictions):
#     """Save predictions to CSV file."""
#     os.makedirs('data', exist_ok=True)
#     output_path = f'data/{city_name}_aqi_forecast.csv'
#     predictions.to_csv(output_path, index=False)
#     logging.info(f"Predictions saved to {output_path}")

# if __name__ == "__main__":
#     cities = {
#         "Karachi": (24.8607, 67.0011),
#         "Lahore": (31.5204, 74.3587),
#         "Islamabad": (33.6844, 73.0479)
#     }

#     for city, (lat, lon) in cities.items():
#         logging.info(f"\n=== Starting forecast for {city} ===")
#         try:
#             # Load historical data for lag features
#             historical_path = f'data/{city}_historical_aq.csv'
#             if not os.path.exists(historical_path):
#                 raise FileNotFoundError(f"Historical data for {city} not found: {historical_path}")
#             historical_df = pd.read_csv(historical_path)
            
#             # Fetch forecast data
#             forecast_df = fetch_forecast_data(lat, lon, city)
            
#             # Create features
#             forecast_df = create_features(forecast_df, historical_df)
            
#             # Load models
#             models = load_models(city)
            
#             # Generate predictions
#             predictions = make_predictions(city, forecast_df, models)
            
#             # Save predictions
#             save_predictions(city, predictions)
            
#             # Show last few rows
#             logging.info(f"Last 5 rows of predictions for {city}:\n{predictions.tail()}")
            
#         except Exception as e:
#             logging.error(f"Forecast processing failed for {city}: {str(e)}")
#             raise

# ----------------------------TRY WITH SHAP -------------------------

# import os
# import warnings
# import logging
# import numpy as np
# import pandas as pd
# import joblib
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry

# # --- Plotting/SHAP ---
# import matplotlib
# matplotlib.use("Agg")  # headless-safe
# import matplotlib.pyplot as plt
# import shap
# warnings.filterwarnings("ignore", category=UserWarning)

# # ---------------- Logging setup for CI/CD ----------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("forecast.log"), logging.StreamHandler()]
# )

# # ---------------- Open-Meteo client ----------------
# def setup_openmeteo_client():
#     """Set up Open-Meteo client with cache and retries."""
#     cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     return openmeteo_requests.Client(session=retry_session)

# # ---------------- Data fetch ----------------
# def fetch_forecast_data(latitude, longitude, city_name):
#     """Fetch 72-hour forecast data from Open-Meteo."""
#     logging.info(f"Fetching forecast data for {city_name}...")
#     openmeteo = setup_openmeteo_client()
#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": [
#             "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
#             "sulphur_dioxide", "ozone", "dust", "uv_index", "us_aqi"
#         ],
#         "forecast_days": 3,
#     }
#     try:
#         responses = openmeteo.weather_api(url, params=params)
#         response = responses[0]
#     except Exception as e:
#         logging.error(f"Error fetching forecast data for {city_name}: {str(e)}")
#         raise

#     hourly = response.Hourly()
#     hourly_data = {
#         "date": pd.date_range(
#             start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#             end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#             freq=pd.Timedelta(seconds=hourly.Interval()),
#             inclusive="left"
#         )
#     }
#     for i, var in enumerate(params["hourly"]):
#         hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

#     df = pd.DataFrame(data=hourly_data)
#     df['city'] = city_name
#     return df

# # ---------------- Feature engineering ----------------
# def create_features(df, historical_df):
#     """Create forecast features using historical data."""
#     # Ensure datetime & correct dtype
#     if 'date' in historical_df.columns:
#         historical_df['date'] = pd.to_datetime(historical_df['date'], utc=True, errors='coerce')
#     if 'us_aqi' in historical_df.columns:
#         historical_df['us_aqi'] = pd.to_numeric(historical_df['us_aqi'], errors='coerce')

#     # Keep last 48 hours of history for lag/moving stats
#     last_48_hours = historical_df.tail(48)[['date', 'us_aqi']].copy()

#     forecast_df = df[['date', 'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
#                       'sulphur_dioxide', 'ozone', 'dust', 'uv_index', 'city']].copy()

#     # Combine to maintain continuity for rolling/lag features
#     combined_df = pd.concat([last_48_hours, forecast_df], ignore_index=True)

#     combined_df['hour'] = combined_df['date'].dt.hour
#     combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
#     combined_df['month'] = combined_df['date'].dt.month

#     combined_df['aqi_lag_24'] = combined_df['us_aqi'].shift(24)
#     combined_df['aqi_lag_48'] = combined_df['us_aqi'].shift(48)
#     combined_df['aqi_rolling_24_mean'] = combined_df['us_aqi'].rolling(window=24, min_periods=1).mean()
#     combined_df['aqi_rolling_24_std'] = combined_df['us_aqi'].rolling(window=24, min_periods=1).std()
#     combined_df['aqi_change_rate'] = (combined_df['us_aqi'] - combined_df['aqi_lag_24']) / combined_df['aqi_lag_24']

#     # Fill missing values (safe for time series inference window)
#     combined_df = combined_df.ffill().bfill()

#     # Keep only the 72-hr forecast window (last 72 rows)
#     forecast_only = combined_df.tail(72).copy()
#     return forecast_only

# # ---------------- Model loading ----------------
# def load_models(city_name):
#     """Load models and scaler for the given city."""
#     models = {}
#     model_dir = "models"
#     try:
#         # Random Forest
#         rf_path = os.path.join(model_dir, f"{city_name}_random_forest.joblib")
#         if os.path.exists(rf_path):
#             models['random_forest'] = joblib.load(rf_path)
#         else:
#             raise FileNotFoundError(f"Random Forest model for {city_name} not found: {rf_path}")

#         # Ridge
#         ridge_path = os.path.join(model_dir, f"{city_name}_ridge.joblib")
#         if os.path.exists(ridge_path):
#             ridge, ridge_scaler = joblib.load(ridge_path)
#             models['ridge'] = {"model": ridge, "scaler": ridge_scaler}
#         else:
#             raise FileNotFoundError(f"Ridge model for {city_name} not found: {ridge_path}")

#         # SVR
#         svr_path = os.path.join(model_dir, f"{city_name}_svr.joblib")
#         if os.path.exists(svr_path):
#             svr, svr_scaler = joblib.load(svr_path)
#             models['svr'] = {"model": svr, "scaler": svr_scaler}
#         else:
#             raise FileNotFoundError(f"SVR model for {city_name} not found: {svr_path}")

#     except Exception as e:
#         logging.error(f"Error loading models for {city_name}: {str(e)}")
#         raise
#     return models

# # ---------------- SHAP helpers ----------------
# def _ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)

# def _save_shap_summary_and_bar(shap_values, features_df, out_prefix: str):
#     """Save beeswarm summary and mean |SHAP| bar plots to PNG."""
#     # Beeswarm
#     plt.figure()
#     shap.summary_plot(shap_values, features=features_df, show=False)
#     plt.tight_layout()
#     plt.savefig(f"{out_prefix}_summary.png", dpi=200, bbox_inches="tight")
#     plt.close()

#     # Bar (mean |SHAP|)
#     plt.figure()
#     shap.summary_plot(shap_values, features=features_df, plot_type="bar", show=False)
#     plt.tight_layout()
#     plt.savefig(f"{out_prefix}_bar.png", dpi=200, bbox_inches="tight")
#     plt.close()

# def explain_with_shap(city_name: str, model_name: str, model_obj, X_df: pd.DataFrame):
#     """
#     Generate SHAP plots for a model and save PNG files.
#     - RandomForest: TreeExplainer
#     - Ridge: LinearExplainer (on scaled X)
#     - SVR: KernelExplainer (on scaled X) with sampling for speed
#     """
#     out_dir = os.path.join("shap_plots", city_name)
#     _ensure_dir(out_dir)
#     out_prefix = os.path.join(out_dir, f"{city_name}_{model_name}")

#     try:
#         if model_name == "random_forest":
#             # Tree-based → fast
#             explainer = shap.TreeExplainer(model_obj)
#             shap_values = explainer.shap_values(X_df)  # returns np.array for regression
#             _save_shap_summary_and_bar(shap_values, X_df, out_prefix)
#             logging.info(f"SHAP plots saved: {out_prefix}_summary.png, {out_prefix}_bar.png")

#         elif model_name == "ridge":
#             ridge = model_obj["model"]
#             scaler = model_obj["scaler"]

#             X_scaled = pd.DataFrame(
#                 scaler.transform(X_df),
#                 index=X_df.index,
#                 columns=X_df.columns
#             )

#             # Use unified API; will pick LinearExplainer
#             explainer = shap.Explainer(ridge, X_scaled)
#             shap_exp = explainer(X_scaled)  # shap.Explanation
#             _save_shap_summary_and_bar(shap_exp.values, X_df, out_prefix)
#             logging.info(f"SHAP plots saved: {out_prefix}_summary.png, {out_prefix}_bar.png")

#         elif model_name == "svr":
#             svr = model_obj["model"]
#             scaler = model_obj["scaler"]

#             X_scaled = pd.DataFrame(
#                 scaler.transform(X_df),
#                 index=X_df.index,
#                 columns=X_df.columns
#             )

#             # ---- Speed-conscious KernelExplainer setup ----
#             # Background (masker) via KMeans on up to 100 samples
#             bg_n = min(100, len(X_scaled))
#             bg_data = shap.kmeans(X_scaled, bg_n)

#             # Evaluation subset to limit compute (up to 200 rows, here 72 by default)
#             eval_n = min(200, len(X_scaled))
#             X_eval = X_scaled.iloc[:eval_n]

#             # Define prediction function operating on scaled space
#             f = lambda X_: svr.predict(X_)

#             explainer = shap.KernelExplainer(f, bg_data)
#             # nsamples="auto" can still be heavy; cap it sensibly
#             shap_values = explainer.shap_values(X_eval, nsamples=100)

#             # For plotting labels we want original (unscaled) features:
#             X_eval_unscaled_like = X_df.iloc[:eval_n].copy()

#             # Save
#             out_prefix_eval = out_prefix + "_eval"
#             _save_shap_summary_and_bar(shap_values, X_eval_unscaled_like, out_prefix_eval)
#             logging.info(f"SHAP plots saved: {out_prefix_eval}_summary.png, {out_prefix_eval}_bar.png")

#         else:
#             logging.warning(f"No SHAP configured for model '{model_name}'. Skipping.")

#     except Exception as e:
#         logging.error(f"Error generating SHAP plots for {city_name} - {model_name}: {str(e)}")

# # ---------------- Prediction ----------------
# def make_predictions(city_name, forecast_df, models):
#     """Generate 72-hour predictions using loaded models and produce SHAP plots."""
#     logging.info(f"Generating predictions for {city_name}...")

#     feature_columns = [
#         'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
#         'sulphur_dioxide', 'ozone', 'dust', 'uv_index',
#         'hour', 'day_of_week', 'month',
#         'aqi_lag_24', 'aqi_lag_48', 'aqi_rolling_24_mean', 'aqi_rolling_24_std',
#         'aqi_change_rate'
#     ]

#     X_df = forecast_df[feature_columns].copy()
#     # Ensure numeric only (should already be numeric but safe)
#     X_df = X_df.apply(pd.to_numeric, errors='coerce')

#     # Fill any remaining NaNs
#     if X_df.isna().any().any():
#         logging.warning(f"NaN values found in forecast features for {city_name}, filling with ffill/bfill...")
#         X_df = X_df.ffill().bfill()

#     predictions = {}
#     for model_name, model in models.items():
#         try:
#             if model_name in ['ridge', 'svr']:
#                 X_scaled = model['scaler'].transform(X_df)
#                 pred = model['model'].predict(X_scaled)
#             else:
#                 pred = model.predict(X_df)

#             predictions[model_name] = pred

#             # ---- SHAP per model ----
#             explain_with_shap(city_name, model_name, model, X_df)

#         except Exception as e:
#             logging.error(f"Error predicting with {model_name} for {city_name}: {str(e)}")
#             raise

#     # Build results
#     results = forecast_df[['date', 'city']].copy()
#     for model_name, pred in predictions.items():
#         results[f'{model_name}_pred'] = pred

#     return results

# # ---------------- Save ----------------
# def save_predictions(city_name, predictions):
#     """Save predictions to CSV file."""
#     os.makedirs('data', exist_ok=True)
#     output_path = f'data/{city_name}_aqi_forecast.csv'
#     predictions.to_csv(output_path, index=False)
#     logging.info(f"Predictions saved to {output_path}")

# # ---------------- Main ----------------
# if __name__ == "__main__":
#     cities = {
#         "Karachi": (24.8607, 67.0011),
#         "Lahore": (31.5204, 74.3587),
#         "Islamabad": (33.6844, 73.0479)
#     }

#     for city, (lat, lon) in cities.items():
#         logging.info(f"\n=== Starting forecast for {city} ===")
#         try:
#             # Load historical data for lag features
#             historical_path = f'data/{city}_historical_aq.csv'
#             if not os.path.exists(historical_path):
#                 raise FileNotFoundError(f"Historical data for {city} not found: {historical_path}")

#             # Parse with correct dtypes
#             historical_df = pd.read_csv(historical_path)
#             if 'date' in historical_df.columns:
#                 historical_df['date'] = pd.to_datetime(historical_df['date'], utc=True, errors='coerce')
#             if 'us_aqi' in historical_df.columns:
#                 historical_df['us_aqi'] = pd.to_numeric(historical_df['us_aqi'], errors='coerce')

#             # Fetch forecast data
#             forecast_raw = fetch_forecast_data(lat, lon, city)

#             # Create features
#             forecast_df = create_features(forecast_raw, historical_df)

#             # Load models
#             models = load_models(city)

#             # Generate predictions (and SHAP plots)
#             predictions = make_predictions(city, forecast_df, models)

#             # Save predictions
#             save_predictions(city, predictions)

#             # Show last few rows
#             logging.info(f"Last 5 rows of predictions for {city}:\n{predictions.tail()}")

#         except Exception as e:
#             logging.error(f"Forecast processing failed for {city}: {str(e)}")
#             raise






import os
import pandas as pd
import numpy as np
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def setup_openmeteo_client():
    """Set up Open-Meteo client with cache and retries."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def fetch_forecast_data(latitude, longitude, city_name):
    """Fetch 72-hour forecast data from Open-Meteo."""
    openmeteo = setup_openmeteo_client()
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                   "sulphur_dioxide", "ozone", "dust", "uv_index", "us_aqi"],
        "forecast_days": 3,
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
def create_features(df, historical_df):
    """Create forecast features using historical data."""
    # Add historical data for lags and moving averages
    last_48_hours = historical_df.tail(48)[['date', 'us_aqi']].copy()
    forecast_df = df[['date', 'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
                     'sulphur_dioxide', 'ozone', 'dust', 'uv_index', 'city']].copy()

    # Convert 'date' to datetime for both DataFrames
    last_48_hours['date'] = pd.to_datetime(last_48_hours['date'], errors='coerce', utc=True)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce', utc=True)

    # Combine historical and forecast data
    combined_df = pd.concat([last_48_hours, forecast_df], ignore_index=True)

    # Feature creation
    combined_df['hour'] = combined_df['date'].dt.hour
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    combined_df['month'] = combined_df['date'].dt.month

    combined_df['aqi_lag_24'] = combined_df['us_aqi'].shift(24)
    combined_df['aqi_lag_48'] = combined_df['us_aqi'].shift(48)
    combined_df['aqi_rolling_24_mean'] = combined_df['us_aqi'].rolling(window=24).mean()
    combined_df['aqi_rolling_24_std'] = combined_df['us_aqi'].rolling(window=24).std()
    combined_df['aqi_change_rate'] = (
        (combined_df['us_aqi'] - combined_df['aqi_lag_24']) /
        combined_df['aqi_lag_24']
    )

    # Fill missing values
    combined_df = combined_df.ffill().bfill()

    # Select only forecast data (last 72 hours)
    forecast_df = combined_df.tail(72).copy()
    return forecast_df


def load_models(city_name):
    """Load models and scaler for the given city."""
    models = {}
    model_dir = "models"

    rf_path = os.path.join(model_dir, f"{city_name}_random_forest.joblib")
    if os.path.exists(rf_path):
        models['random_forest'] = joblib.load(rf_path)

    ridge_path = os.path.join(model_dir, f"{city_name}_ridge.joblib")
    if os.path.exists(ridge_path):
        ridge, ridge_scaler = joblib.load(ridge_path)
        models['ridge'] = {"model": ridge, "scaler": ridge_scaler}

    svr_path = os.path.join(model_dir, f"{city_name}_svr.joblib")
    if os.path.exists(svr_path):
        svr, svr_scaler = joblib.load(svr_path)
        models['svr'] = {"model": svr, "scaler": svr_scaler}

    return models

def make_predictions(city_name, forecast_df, models):
    """Generate 72-hour predictions using loaded models."""
    feature_columns = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
                      'sulphur_dioxide', 'ozone', 'dust', 'uv_index', 
                      'hour', 'day_of_week', 'month', 
                      'aqi_lag_24', 'aqi_lag_48', 'aqi_rolling_24_mean', 'aqi_rolling_24_std', 
                      'aqi_change_rate']
    
    X = forecast_df[feature_columns].select_dtypes(include=[np.number])
    X = X.ffill().bfill()

    predictions = {}
    for model_name, model in models.items():
        if model_name in ['ridge', 'svr']:
            X_scaled = model['scaler'].transform(X)
            pred = model['model'].predict(X_scaled)
        else:
            pred = model.predict(X)
        predictions[model_name] = pred
    
    results = forecast_df[['date', 'city']].copy()
    for model_name, pred in predictions.items():
        results[f'{model_name}_pred'] = pred
    return results, X

def save_predictions(city_name, predictions):
    """Save predictions to CSV file."""
    os.makedirs('data', exist_ok=True)
    predictions.to_csv(f'data/{city_name}_aqi_forecast.csv', index=False)

def save_shap_plots(city_name, models, X):
    """Generate and save SHAP plots for each model without logging."""
    os.makedirs("shap_plots", exist_ok=True)
    for model_name, model in models.items():
        try:
            if model_name == 'random_forest':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                explainer = shap.Explainer(model['model'], model['scaler'].transform(X))
                shap_values = explainer(model['scaler'].transform(X))
            
            plt.figure()
            shap.summary_plot(shap_values, X if model_name == 'random_forest' else model['scaler'].transform(X), show=False)
            plt.tight_layout()
            plt.savefig(f"shap_plots/{city_name}_{model_name}_summary.png", dpi=200, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(shap_values, X if model_name == 'random_forest' else model['scaler'].transform(X), plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f"shap_plots/{city_name}_{model_name}_bar.png", dpi=200, bbox_inches="tight")
            plt.close()
        except:
            pass

if __name__ == "__main__":
    cities = {
        "Karachi": (24.8607, 67.0011),
        "Lahore": (31.5204, 74.3587),
        "Islamabad": (33.6844, 73.0479)
    }

    for city, (lat, lon) in cities.items():
        historical_df = pd.read_csv(f'data/{city}_historical_aq.csv')
        forecast_df = fetch_forecast_data(lat, lon, city)
        forecast_df = create_features(forecast_df, historical_df)
        models = load_models(city)
        predictions, X = make_predictions(city, forecast_df, models)
        save_predictions(city, predictions)
        save_shap_plots(city, models, X)
