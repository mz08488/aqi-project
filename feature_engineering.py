# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import joblib
# import os

# def calculate_aqi(pollutants):
#     # Simplified AQI calculation (replace with proper formula)
#     pm25 = pollutants['pm2_5']
#     pm10 = pollutants['pm10']
#     no2 = pollutants['nitrogen_dioxide']
#     so2 = pollutants['sulphur_dioxide']
#     o3 = pollutants['ozone']
#     co = pollutants['carbon_monoxide']
    
#     # Weighted average
#     aqi = (pm25*0.3 + pm10*0.2 + no2*0.15 + so2*0.15 + o3*0.1 + co*0.1)
#     return aqi

# def create_features(df):
#     # Calculate AQI
#     df['aqi'] = df.apply(lambda x: calculate_aqi(x), axis=1)
    
#     # Time-based features
#     df['hour'] = df['date'].dt.hour
#     df['day_of_week'] = df['date'].dt.dayofweek
#     df['month'] = df['date'].dt.month
    
#     # Lag features
#     df['aqi_lag_24'] = df['aqi'].shift(24)
#     df['aqi_lag_48'] = df['aqi'].shift(48)
    
#     # Rolling features
#     df['aqi_rolling_24_mean'] = df['aqi'].rolling(window=24).mean()
#     df['aqi_rolling_24_std'] = df['aqi'].rolling(window=24).std()
    
#     # Change rate
#     df['aqi_change_rate'] = (df['aqi'] - df['aqi_lag_24']) / df['aqi_lag_24']
    
#     # Drop rows with NaN values from lag features
#     df = df.dropna()
    
#     return df

# def prepare_data(city_name):
#     # Load data
#     df = pd.read_csv(f'data/{city_name}_air_quality.csv', parse_dates=['date'])
    
#     # Create features
#     df = create_features(df)
    
#     # Split into features and target
#     features = df.drop(columns=['date', 'city', 'aqi'])
#     target = df['aqi']
    
#     # Scale features
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
    
#     # Save scaler
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(scaler, f'models/{city_name}_scaler.joblib')
    
#     return scaled_features, target, df

# # def prepare_data(city_name):
# #     # Load data
# #     df = pd.read_csv(f'data/{city_name}_air_quality.csv', parse_dates=['date'])
    
# #     # Create features
# #     df = create_features(df)
    
# #     # Split FIRST then scale
# #     train_df = df.iloc[:int(0.8*len(df))]  # 80% train
# #     test_df = df.iloc[int(0.8*len(df)):]   # 20% test
    
# #     # Scale features
# #     scaler = StandardScaler()
# #     X_train = scaler.fit_transform(train_df.drop(columns=['date', 'city', 'aqi']))
# #     y_train = train_df['aqi']
    
# #     X_test = scaler.transform(test_df.drop(columns=['date', 'city', 'aqi']))
# #     y_test = test_df['aqi']
    
# #     # Save scaler
# #     os.makedirs('models', exist_ok=True)
# #     joblib.dump(scaler, f'models/{city_name}_scaler.joblib')
    
# #     return X_train, X_test, y_train, y_test, df

# if __name__ == "__main__":
#     for city in ["Karachi", "Lahore", "Islamabad"]:
#         print(f"Processing features for {city}...")
#         prepare_data(city)




#  ---------------- final try -------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# def calculate_aqi(row):
#     # Simplified AQI calculation (replace with proper formula)
#     pm25 = row['pm2_5']
#     pm10 = row['pm10']
#     no2 = row['nitrogen_dioxide']
#     so2 = row['sulphur_dioxide']
#     o3 = row['ozone']
#     co = row['carbon_monoxide']
    
#     # Weighted average
#     aqi = (pm25*0.3 + pm10*0.2 + no2*0.15 + so2*0.15 + o3*0.1 + co*0.1)
#     return aqi

def create_features(df):
    # Calculate AQI
    # df['aqi'] = df.apply(calculate_aqi, axis=1)
    
    # Time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Lag features
    df['aqi_lag_24'] = df['us_aqi'].shift(24)
    df['aqi_lag_48'] = df['us_aqi'].shift(48)
    
    # Rolling features
    df['aqi_rolling_24_mean'] = df['us_aqi'].rolling(window=24).mean()
    df['aqi_rolling_24_std'] = df['us_aqi'].rolling(window=24).std()
    
    # Change rate
    df['aqi_change_rate'] = (df['us_aqi'] - df['aqi_lag_24']) / df['aqi_lag_24']
    # apply ffill to all df columns
    df.fillna(method='ffill', inplace=True)
    # Drop rows with NaN values from lag features
    df = df.dropna()
    
    return df

def prepare_data(city_name):
    # Load data
    df = pd.read_csv(f'data/{city_name}_historical_aq.csv', parse_dates=['date'])
    
    # Create features
    df = create_features(df)
    
    # Split into features and target
    features = df.drop(columns=['date', 'city', 'us_aqi',"aqi'])"])
    target = df['us_aqi']
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/{city_name}_scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df

if __name__ == "__main__":
    for city in ["Karachi", "Lahore", "Islamabad"]:
        print(f"Processing features for {city}...")
        prepare_data(city)