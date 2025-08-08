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


import pandas as pd
import numpy as np
import joblib
from feature_engineering import prepare_data

def load_models(city_name):
    models = {}
    
    # Load models
    models['random_forest'] = joblib.load(f'models/{city_name}_random_forest.joblib')
    models['ridge'] = joblib.load(f'models/{city_name}_ridge.joblib')
    models['svr'] = joblib.load(f'models/{city_name}_svr.joblib')
    
    # Load Scaler
    scaler = joblib.load(f'models/{city_name}_scaler.joblib')
    
    return models, scaler

def make_predictions(city_name):
    # Prepare data (this will include forecast data)
    X, _, _, _, df = prepare_data(city_name)
    
    # Load models
    models, _ = load_models(city_name)
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = pred
    
    # Create DataFrame with predictions
    results = df[['date', 'city']].copy()
    
    # Ensure we only keep rows that match our predictions
    results = results.iloc[-len(predictions['random_forest']):].copy()
    
    # Add predictions
    for name, pred in predictions.items():
        results[f'{name}_pred'] = pred
    
    # Save predictions
    results.to_csv(f'data/{city_name}_predictions.csv', index=False)
    
    return results

if __name__ == "__main__":
    for city in ["Karachi", "Lahore", "Islamabad"]:
        print(f"Making predictions for {city}...")
        predictions = make_predictions(city)
        print(predictions.tail())