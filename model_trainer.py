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


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from feature_engineering import prepare_data

def train_models(city_name):
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(city_name)
    
    # Dictionary to store models and metrics
    models = {}
    metrics = {}
    
    # 1. Random Forest
    print(f"Training Random Forest for {city_name}...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # 2. Ridge Regression
    print(f"Training Ridge Regression for {city_name}...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['ridge'] = ridge
    
    # 3. Support Vector Regression
    print(f"Training SVR for {city_name}...")
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['svr'] = svr
    
    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics[name] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, f'models/{city_name}_{name}.joblib')
    
    return models, metrics

if __name__ == "__main__":
    for city in ["Karachi", "Lahore", "Islamabad"]:
        print(f"Training models for {city}...")
        models, metrics = train_models(city)
        print(f"Metrics for {city}:")
        for model_name, metric in metrics.items():
            print(f"{model_name}: RMSE={metric['rmse']:.2f}, MAE={metric['mae']:.2f}, R2={metric['r2']:.2f}")