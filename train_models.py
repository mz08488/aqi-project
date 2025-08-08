# import os
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# CITIES = ["Karachi", "Lahore", "Islamabad"]
# FEATURES = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]

# os.makedirs("models", exist_ok=True)

# def train_and_save(city):
#     path = f"data/historical/aqi_features_{city.lower()}.csv"
#     df = pd.read_csv(path)
#     df.dropna(subset=["target_aqi"], inplace=True)
#     X = df[FEATURES]
#     y = df["target_aqi"]

#     models = {
#         "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "ridge": Ridge(alpha=1.0)
#     }

#     for name, model in models.items():
#         model.fit(X, y)
#         joblib.dump(model, f"models/{city.lower()}_{name}.pkl")

# if __name__ == "__main__":
#     for city in CITIES:
#         train_and_save(city)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from math import sqrt
# import joblib
# import os

# CITIES = ["Karachi", "Lahore", "Islamabad"]
# FEATURES = ["hour", "day", "month", "pm10", "pm2_5", "aqi_change"]
# TARGET = 'target_aqi'

# os.makedirs("models", exist_ok=True)

# def train_and_evaluate(city):
#     print(f"\n🏙️ Training model for {city}...")
#     path = os.path.join("data", "historical", f"aqi_features_{city.lower()}.csv")
#     df = pd.read_csv(path)

#     df.dropna(subset=[TARGET], inplace=True)

#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'])
#         df['day'] = df['date'].dt.day
#         df['month'] = df['date'].dt.month
#         df['year'] = df['date'].dt.year
#         df['dayofweek'] = df['date'].dt.dayofweek

#     # ✅ Check for missing features
#     existing_features = [f for f in FEATURES if f in df.columns]
#     missing = set(FEATURES) - set(existing_features)
#     if missing:
#         print(f"⚠️ Warning: Missing features in {city}: {missing}")

#     X = df[existing_features]
#     y = df[TARGET]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     models = {
#         "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "ridge": Ridge(alpha=1.0)
#     }

#     for name, model in models.items():
#         print(f"\n🔧 Training {name}...")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         mae = mean_absolute_error(y_test, y_pred)
        
#         rmse = sqrt(mean_squared_error(y_test, y_pred))

#         r2 = r2_score(y_test, y_pred)

#         print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")

#         model_path = os.path.join("models", f"{city.lower()}_{name}.pkl")
#         joblib.dump(model, model_path)
#         print(f"✅ Saved model to {model_path}")


# if __name__ == '__main__':
#     for city in CITIES:
#         train_and_evaluate(city)



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from model_registry import save_model

models = {
    "random_forest": RandomForestRegressor(),
    "ridge": Ridge(),
    "decision_tree": DecisionTreeRegressor(),
    "svr": SVR()
}

def train_and_save(city):
    df = pd.read_csv(f"data/{city}_past_data.csv", parse_dates=['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['aqi_estimate'] = df[['pm10', 'pm2_5']].mean(axis=1)
    X = df[['pm10', 'pm2_5', 'hour', 'day', 'month']]
    y = df['aqi_estimate']

    for name, model in models.items():
        model.fit(X, y)
        save_model(model, f"models/{city}_{name}.pkl")

if __name__ == '__main__':
    for city in ["karachi", "lahore", "islamabad"]:
        train_and_save(city)

