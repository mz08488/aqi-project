# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# import tensorflow as tf
# from feature_engineering import prepare_data

# def train_models(city_name):
#     # Prepare data
#     X, y, _ = prepare_data(city_name)
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, shuffle=False)
    
#     # Dictionary to store models and metrics
#     models = {}
#     metrics = {}
    
#     # 1. Random Forest
#     print(f"Training Random Forest for {city_name}...")
#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#     models['random_forest'] = rf
    
#     # 2. Ridge Regression
#     print(f"Training Ridge Regression for {city_name}...")
#     ridge = Ridge(alpha=1.0)
#     ridge.fit(X_train, y_train)
#     models['ridge'] = ridge
    
#     # 3. Neural Network
#     print(f"Training Neural Network for {city_name}...")
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
    
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#     models['neural_network'] = model
    
#     # Evaluate models
#     for name, model in models.items():
#         if name == 'neural_network':
#             y_pred = model.predict(X_test).flatten()
#         else:
#             y_pred = model.predict(X_test)
        
#         metrics[name] = {
#             'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#             'mae': mean_absolute_error(y_test, y_pred),
#             'r2': r2_score(y_test, y_pred)
#         }
    
#     # Save models
#     for name, model in models.items():
#         if name == 'neural_network':
#             model.save(f'models/{city_name}_{name}.h5')
#         else:
#             joblib.dump(model, f'models/{city_name}_{name}.joblib')
    
#     return models, metrics

# if __name__ == "__main__":
#     for city in ["Karachi", "Lahore", "Islamabad"]:
#         print(f"Training models for {city}...")
#         models, metrics = train_models(city)
#         print(f"Metrics for {city}:")
#         for model_name, metric in metrics.items():
#             print(f"{model_name}: RMSE={metric['rmse']:.2f}, MAE={metric['mae']:.2f}, R2={metric['r2']:.2f}")



#----------------- final try ------------------
import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set up logging for CI/CD
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(model, X_train, y_train, X_test, y_test, scaler=None):
    """Helper function to train a model and compute metrics for a single fold."""
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, rmse, mae, r2

def train_or_retrain_model(city_name, csv_path, n_splits=5):
    """Train or retrain models for a given city using TimeSeriesSplit."""
    logging.info(f"Processing {city_name} with data from {csv_path}")

    # Validate CSV file
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV {csv_path}: {str(e)}")
        raise

    if "us_aqi" not in df.columns:
        logging.error(f"'us_aqi' column not found in {csv_path}")
        raise ValueError(f"'us_aqi' column not found in {csv_path}")

    X = df.drop(columns=["us_aqi"]).select_dtypes(include=[np.number])
    y = df["us_aqi"]

    if X.empty:
        logging.error(f"No numeric features found in {csv_path}")
        raise ValueError(f"No numeric features found in {csv_path}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = {}
    metrics = {}

    # Model configurations
    model_configs = [
        {
            "name": "random_forest",
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "use_scaler": False
        },
        {
            "name": "ridge",
            "model": Ridge(alpha=1.0),
            "use_scaler": True
        },
        {
            "name": "svr",
            "model": SVR(kernel="rbf", C=100, gamma=0.1),
            "use_scaler": True
        }
    ]

    for config in model_configs:
        model_name = config["name"]
        model = config["model"]
        use_scaler = config["use_scaler"]
        model_path = os.path.join(MODEL_DIR, f"{city_name}_{model_name}.joblib")
        metrics_path = os.path.join(MODEL_DIR, f"{city_name}_{model_name}_metrics.json")

        logging.info(f"Training {model_name} for {city_name}...")

        fold_rmses, fold_maes, fold_r2s = [], [], []
        scaler = StandardScaler() if use_scaler else None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            trained_model, rmse, mae, r2 = train_model(model, X_train, y_train, X_test, y_test, scaler)
            fold_rmses.append(rmse)
            fold_maes.append(mae)
            fold_r2s.append(r2)
            logging.info(f"{model_name} Fold {fold} RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        # Compute average metrics
        avg_rmse = np.mean(fold_rmses)
        avg_mae = np.mean(fold_maes)
        avg_r2 = np.mean(fold_r2s)
        logging.info(f"{model_name} Average RMSE: {avg_rmse:.2f}, MAE: {avg_mae:.2f}, R2: {avg_r2:.2f}")

        # Retrain on full dataset
        if use_scaler:
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            joblib.dump((model, scaler), model_path)
        else:
            model.fit(X, y)
            joblib.dump(model, model_path)

        # Save metrics
        metrics[model_name] = {
            "avg_rmse": avg_rmse,
            "avg_mae": avg_mae,
            "avg_r2": avg_r2
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics[model_name], f, indent=2)

        # Store model (and scaler if applicable)
        if use_scaler:
            models[model_name] = {"model": model, "scaler": scaler}
        else:
            models[model_name] = model

    return models, metrics

if __name__ == "__main__":
    city_csv_map = {
        "Karachi": "data/Karachi_historical_aq.csv",
        "Lahore": "data/Lahore_historical_aq.csv",
        "Islamabad": "data/Islamabad_historical_aq.csv"
    }

    for city, csv_path in city_csv_map.items():
        logging.info(f"\n=== Training/Retraining Models for {city} ===")
        try:
            models, metrics = train_or_retrain_model(city, csv_path)
            logging.info(f"Metrics for {city}:")
            for model_name, metric in metrics.items():
                logging.info(
                    f"{model_name}: avg_rmse={metric['avg_rmse']:.2f}, "
                    f"avg_mae={metric['avg_mae']:.2f}, avg_r2={metric['avg_r2']:.2f}"
                )
        except Exception as e:
            logging.error(f"Failed to process {city}: {str(e)}")
            raise