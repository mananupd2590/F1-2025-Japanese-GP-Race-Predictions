#!/usr/bin/env python
# coding: utf-8


import os
import fastf1

cache_dir = 'cache'

# Automatically create the directory if it doesn't exist yet
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)


# In[5]:


import fastf1
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Enable cache for faster data retrieval
fastf1.Cache.enable_cache('cache')

def get_canadian_gp_results(year):
    """
    Fetches race results for the Canadian GP for the given year.
    Assumes that the event name contains 'Canadian' (case insensitive).
    Returns a DataFrame with DriverNumber, Abbreviation, and Position.
    """
    # Get the season schedule
    schedule = fastf1.get_event_schedule(year)
    
    # Filter to find the Canadian GP (adjust the keyword if needed)
    event = schedule[schedule['EventName'].str.contains("Canadian", case=False, na=False)]
    
    if not event.empty:
        round_number = event.iloc[0]['RoundNumber']
        session = fastf1.get_session(year, round_number, 'R')  # 'R' for race session
        session.load()
        results = session.results[['DriverNumber', 'Abbreviation', 'Position']]
        results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
        return results
    else:
        print(f"No Canadian GP found for {year}")
        return pd.DataFrame()

# Fetch race results for 2022 (dry) and 2023 (rain)
results_2022 = get_canadian_gp_results(2022)
results_2023 = get_canadian_gp_results(2023)

# Rename the Position columns for clarity
results_2022 = results_2022.rename(columns={'Position': 'Position_dry'})
results_2023 = results_2023.rename(columns={'Position': 'Position_rain'})

# Merge results on DriverNumber and Abbreviation (to align drivers)
merged = pd.merge(results_2022, results_2023, on=['DriverNumber', 'Abbreviation'], how='inner')

# Calculate the wet driver score as the ratio of dry position to rain position
# (A score > 1 indicates relatively better performance in wet conditions)
merged['WetDriverScore'] = merged['Position_dry'] / merged['Position_rain']

# Display the results
print("Wet Driver Scores based on Canadian GP performance:")
print(merged[['DriverNumber', 'Abbreviation', 'Position_dry', 'Position_rain', 'WetDriverScore']])


# In[6]:


canada_merged = pd.DataFrame({
    'DriverNumber': [...],
    'Abbreviation': [...],
    'Position_dry': [...],
    'Position_rain': [...],
    'WetDriverScore': [...]
})


# In[7]:


wet_score_dict = dict(zip(canada_merged['Abbreviation'], canada_merged['WetDriverScore']))


# In[8]:


import requests
import datetime
import json

def fetch_race_day_forecast(api_key, city="Suzuka", target_datetime_str="2025-04-06 14:00:00"):
    """
    Fetches the forecast for the given city from OpenWeatherMap and returns the forecast
    entry closest to the target datetime.
    
    Parameters:
      - api_key: Your OpenWeatherMap API key.
      - city: City name (default: "Suzuka").
      - target_datetime_str: Target datetime as a string in "YYYY-MM-DD HH:MM:SS" format.
    
    Returns:
      A dictionary with:
        - 'date': datetime object of the forecast entry,
        - 'rain_probability': probability of precipitation (0.0 to 1.0),
        - 'temperature_c': Temperature in Celsius,
        - 'humidity': Humidity percentage.
      Returns None if no data is available.
    
    Note: The free forecast API only covers the next 5 days. For forecasts beyond that, you
          must use a paid service or a different endpoint.
    """
    target_datetime = datetime.datetime.strptime(target_datetime_str, "%Y-%m-%d %H:%M:%S")
    
    # This URL calls the 5-day/3-hour forecast endpoint (free tier)
    # For dates in 2025, this will not work; this is just for demonstration.
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code} {response.reason}")
        return None
    
    data = response.json()
    forecast_list = data.get("list", [])
    if not forecast_list:
        print("No forecast data available.")
        return None

    best_entry = None
    min_diff = None
    for entry in forecast_list:
        dt_str = entry.get("dt_txt")  # e.g. "2025-09-21 15:00:00"
        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        diff = abs((dt_obj - target_datetime).total_seconds())
        if min_diff is None or diff < min_diff:
            min_diff = diff
            best_entry = entry

    if not best_entry:
        print("No forecast entry found near the target datetime; using last available forecast.")
        best_entry = forecast_list[-1]
    
    forecast_dt = datetime.datetime.strptime(best_entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
    rain_probability = best_entry.get("pop", 0.0)  # Probability of precipitation
    temperature_c = best_entry["main"]["temp"]
    humidity = best_entry["main"]["humidity"]
    
    forecast = {
        "date": forecast_dt,
        "rain_probability": rain_probability,
        "temperature_c": temperature_c,
        "humidity": humidity
    }
    return forecast

if __name__ == "__main__":
    # Use the API key provided in your welcome email
    api_key = "Your_API_Key"
    target_datetime_str = "2025-04-06 14:00:00"  # Race day: April 6, 2025 at 14:00:00
    forecast = fetch_race_day_forecast(api_key, city="Suzuka", target_datetime_str=target_datetime_str)
    
    if forecast:
        print("✅ Forecast for Race Day at Suzuka:")
        print(f"Date/Time: {forecast['date']}")
        print(f"Rain Probability: {forecast['rain_probability']}")
        print(f"Temperature (°C): {forecast['temperature_c']}")
        print(f"Humidity: {forecast['humidity']}%")
    else:
        print("No valid forecast could be retrieved. For long-term forecasts, consider using an extended API service.")


# In[9]:


import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# ---------------------------
# Setup FastF1 cache
# ---------------------------
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}")
fastf1.Cache.enable_cache(cache_dir)

# ---------------------------
# Define Drivers for 2025 Season
# ---------------------------
drivers_2025 = [
    # Ferrari
    {'DriverNumber': 16, 'Abbreviation': 'LEC', 'FullName': 'Charles Leclerc', 'Team': 'Ferrari'},
    {'DriverNumber': 44, 'Abbreviation': 'HAM', 'FullName': 'Lewis Hamilton', 'Team': 'Ferrari'},
    # Mercedes-AMG Petronas
    {'DriverNumber': 63, 'Abbreviation': 'RUS', 'FullName': 'George Russell', 'Team': 'Mercedes'},
    {'DriverNumber': 72, 'Abbreviation': 'ANT', 'FullName': 'Andrea Kimi Antonelli', 'Team': 'Mercedes'},
    # Red Bull Racing
    {'DriverNumber': 1, 'Abbreviation': 'VER', 'FullName': 'Max Verstappen', 'Team': 'Red Bull Racing'},
    {'DriverNumber': 40, 'Abbreviation': 'LAW', 'FullName': 'Liam Lawson', 'Team': 'Red Bull Racing'},
    # McLaren
    {'DriverNumber': 4, 'Abbreviation': 'NOR', 'FullName': 'Lando Norris', 'Team': 'McLaren'},
    {'DriverNumber': 81, 'Abbreviation': 'PIA', 'FullName': 'Oscar Piastri', 'Team': 'McLaren'},
    # Aston Martin
    {'DriverNumber': 14, 'Abbreviation': 'ALO', 'FullName': 'Fernando Alonso', 'Team': 'Aston Martin'},
    {'DriverNumber': 18, 'Abbreviation': 'STR', 'FullName': 'Lance Stroll', 'Team': 'Aston Martin'},
    # Alpine
    {'DriverNumber': 10, 'Abbreviation': 'GAS', 'FullName': 'Pierre Gasly', 'Team': 'Alpine'},
    {'DriverNumber': 5, 'Abbreviation': 'DOO', 'FullName': 'Jack Doohan', 'Team': 'Alpine'},
    # Williams
    {'DriverNumber': 23, 'Abbreviation': 'ALB', 'FullName': 'Alexander Albon', 'Team': 'Williams'},
    {'DriverNumber': 55, 'Abbreviation': 'SAI', 'FullName': 'Carlos Sainz Jr.', 'Team': 'Williams'},
    # Haas
    {'DriverNumber': 31, 'Abbreviation': 'OCO', 'FullName': 'Esteban Ocon', 'Team': 'Haas F1 Team'},
    {'DriverNumber': 87, 'Abbreviation': 'BEA', 'FullName': 'Oliver Bearman', 'Team': 'Haas F1 Team'},
    # Kick Sauber
    {'DriverNumber': 27, 'Abbreviation': 'HUL', 'FullName': 'Nico Hülkenberg', 'Team': 'Kick Sauber'},
    {'DriverNumber': 50, 'Abbreviation': 'BOR', 'FullName': 'Gabriel Bortoleto', 'Team': 'Kick Sauber'},
    # Visa Cash App Racing Bulls (VCARB)
    {'DriverNumber': 22, 'Abbreviation': 'TSU', 'FullName': 'Yuki Tsunoda', 'Team': 'VCARB'},
    {'DriverNumber': 41, 'Abbreviation': 'HAD', 'FullName': 'Isack Hadjar', 'Team': 'VCARB'}
]
drivers_df = pd.DataFrame(drivers_2025)

# ---------------------------
# Functions to Fetch Canadian GP Results and Compute Wet Driver Score
# ---------------------------
def get_canadian_gp_results(year):
    """
    Fetch the Canadian GP race results for a given year using FastF1.
    Looks for the event where "Canadian" appears in the event name.
    Returns a DataFrame with DriverNumber, Abbreviation, and Position.
    """
    schedule = fastf1.get_event_schedule(year)
    event = schedule[schedule['EventName'].str.contains("Canadian", case=False, na=False)]
    if not event.empty:
        round_number = event.iloc[0]['RoundNumber']
        session = fastf1.get_session(year, round_number, 'R')
        session.load()
        results = session.results[['DriverNumber', 'Abbreviation', 'Position']]
        results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
        return results
    else:
        print(f"No Canadian GP found for {year}")
        return pd.DataFrame()

# Fetch results for 2022 (dry) and 2023 (rain)
results_2022 = get_canadian_gp_results(2022)
results_2023 = get_canadian_gp_results(2023)
if results_2022.empty or results_2023.empty:
    print("Insufficient Canadian GP data; please check the event names or data availability.")
else:
    # Rename positions for clarity
    results_2022 = results_2022.rename(columns={'Position': 'Position_dry'})
    results_2023 = results_2023.rename(columns={'Position': 'Position_rain'})
    # Merge on DriverNumber and Abbreviation
    canada_merged = pd.merge(results_2022, results_2023, on=['DriverNumber', 'Abbreviation'], how='inner')
    # Calculate WetDriverScore = Position_dry / Position_rain (higher score indicates relatively better wet performance)
    canada_merged['WetDriverScore'] = canada_merged['Position_dry'] / canada_merged['Position_rain']
    # Create a dictionary mapping driver Abbreviation to WetDriverScore
    wet_score_dict = dict(zip(canada_merged['Abbreviation'], canada_merged['WetDriverScore']))
    print("Wet Driver Scores based on Canadian GP:")
    print(canada_merged[['DriverNumber', 'Abbreviation', 'WetDriverScore']])

# ---------------------------
# Fetch Suzuka Weather Forecast for Race Day (April 6, 2025 at 14:00:00)
# ---------------------------
def fetch_race_day_forecast(api_key, city="Suzuka", target_datetime_str="2025-04-06 14:00:00"):
    """
    Fetch the 5-day/3-hour forecast from OpenWeatherMap for a target datetime.
    Returns a dict with rain_probability, temperature (°C), and humidity.
    Note: Free API only covers 5 days; this is for demonstration.
    """
    target_datetime = datetime.datetime.strptime(target_datetime_str, "%Y-%m-%d %H:%M:%S")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code} {response.reason}")
        return None
    data = response.json()
    forecast_list = data.get("list", [])
    if not forecast_list:
        print("No forecast data available.")
        return None
    
    best_entry = None
    min_diff = None
    for entry in forecast_list:
        dt_obj = datetime.datetime.strptime(entry.get("dt_txt"), "%Y-%m-%d %H:%M:%S")
        diff = abs((dt_obj - target_datetime).total_seconds())
        if min_diff is None or diff < min_diff:
            min_diff = diff
            best_entry = entry
            
    if not best_entry:
        best_entry = forecast_list[-1]
    
    forecast_dt = datetime.datetime.strptime(best_entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
    rain_probability = best_entry.get("pop", 0.0)
    temperature_c = best_entry["main"]["temp"]
    humidity = best_entry["main"]["humidity"]
    
    forecast = {
        "date": forecast_dt,
        "rain_probability": rain_probability,
        "temperature_c": temperature_c,
        "humidity": humidity
    }
    return forecast

# Replace with your actual OpenWeatherMap API key
owm_api_key = "Your_API_Key"
weather_forecast = fetch_race_day_forecast(owm_api_key, city="Suzuka", target_datetime_str="2025-04-06 14:00:00")
if weather_forecast:
    print("\nSuzuka Weather Forecast for 2025-04-06 14:00:00:")
    print(weather_forecast)
else:
    # Fallback values if forecast unavailable (simulate rain conditions)
    weather_forecast = {"rain_probability": 0.8, "temperature_c": 22.0, "humidity": 85}
    print("\nUsing fallback weather values for Suzuka.")

# ---------------------------
# Load Historical Race Data for Training (2022-2024)
# ---------------------------
seasons = [2022, 2023, 2024]
race_data = []
print("\nFetching historical F1 race data...")
for season in seasons:
    for rnd in range(1, 23):  # Try rounds 1 to 22
        try:
            session = fastf1.get_session(season, rnd, 'R')
            session.load()
            res = session.results[['DriverNumber', 'Position', 'GridPosition']]
            res['Season'] = season
            res['RaceNumber'] = rnd
            res['Circuit'] = session.event['EventName']
            race_data.append(res)
            print(f"Loaded {season} Race {rnd}: {session.event['EventName']}")
        except Exception as e:
            # Skip if data not available
            continue

combined_data = pd.concat(race_data)
combined_data['Position'] = pd.to_numeric(combined_data['Position'], errors='coerce').fillna(20)
combined_data['GridPosition'] = pd.to_numeric(combined_data['GridPosition'], errors='coerce').fillna(20)

# Merge with drivers info
combined_data['DriverNumber'] = combined_data['DriverNumber'].astype(int)
full_data = pd.merge(combined_data, drivers_df, on='DriverNumber', how='left')

# ---------------------------
# Feature Engineering
# ---------------------------
# Experience: number of races per driver
full_data['Experience'] = full_data.groupby('DriverNumber').cumcount() + 1

# For historical data, assume dry weather (WeatherCondition=0, WetPerformanceFactor=1.0)
full_data['WeatherCondition'] = 0   # 0 = dry, 1 = rain
full_data['WetPerformanceFactor'] = 1.0

# Weight recent seasons higher (2022 -> 1, 2023 -> 2, 2024 -> 3)
weight_map = {2022: 1, 2023: 2, 2024: 3}
full_data['SampleWeight'] = full_data['Season'].map(weight_map)

# ---------------------------
# Encode categorical features (Team)
# ---------------------------
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoded = encoder.fit_transform(full_data[['Team']])
team_encoded_df = pd.DataFrame(team_encoded, columns=encoder.get_feature_names_out(['Team']))

# Prepare training features and target variable
features = pd.concat([
    full_data[['GridPosition', 'Experience', 'WeatherCondition', 'WetPerformanceFactor']],
    team_encoded_df,
], axis=1)
target = full_data['Position']
sample_weight = full_data['SampleWeight']

# ---------------------------
# Train the Model
# ---------------------------
print("\nTraining RandomForest model with sample weights...")
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(features, target, sample_weight=sample_weight)

# Display feature importances (top 10)
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 10 features:")
print(feature_importance.head(10))

# ---------------------------
# Predict Japanese GP (Suzuka 2025) Results with Weather & Wet Driver Score
# ---------------------------
print("\nPredicting Japanese GP (Suzuka) 2025 Results...")

prediction_rows = []
for driver in drivers_2025:
    # For each driver, simulate a qualifying grid position (e.g., based on historical team average)
    past_races = full_data[full_data['DriverNumber'] == driver['DriverNumber']]
    experience = len(past_races)
    team_avg_grid = past_races['GridPosition'].mean() if not past_races.empty else 10
    
    # Simulate a grid position with randomness
    simulated_grid = np.clip(np.random.normal(team_avg_grid, 2), 1, 20)
    
    # Get wet driver score from our dictionary (default to 1.0 if not found)
    wet_score = wet_score_dict.get(driver['Abbreviation'], 1.0)
    
    prediction_rows.append({
        'Driver': driver['FullName'],
        'Abbreviation': driver['Abbreviation'],
        'Team': driver['Team'],
        'GridPosition': simulated_grid,
        'Experience': experience,
        # For the race, set weather features based on fetched forecast:
        'WeatherCondition': 1 if weather_forecast['rain_probability'] > 0.5 else 0,
        # Incorporate wet driver performance: if it's a wet race, multiply by wet_score
        'WetPerformanceFactor': wet_score,
    })

test_df = pd.DataFrame(prediction_rows)

# Add weather parameters (you could include Temperature and Humidity if desired)
test_df['RainProbability'] = weather_forecast['rain_probability']
test_df['TemperatureC'] = weather_forecast['temperature_c']
test_df['Humidity'] = weather_forecast['humidity']

# Prepare test features (selecting columns that match training)
required_cols = ['GridPosition', 'Experience', 'WeatherCondition', 'WetPerformanceFactor']
test_features = test_df[required_cols].copy()

# If your model was trained with team encoding, add that as well.
team_encoded_test = encoder.transform(test_df[['Team']])
team_encoded_test_df = pd.DataFrame(team_encoded_test, columns=encoder.get_feature_names_out(['Team']))
test_features = pd.concat([test_features.reset_index(drop=True), team_encoded_test_df.reset_index(drop=True)], axis=1)

# Ensure test features include all training columns
for col in features.columns:
    if col not in test_features.columns:
        test_features[col] = 0
test_features = test_features[features.columns]

# Predict finishing positions
predictions = model.predict(test_features)
test_df['PredictedPosition'] = predictions

# Sort drivers by predicted finishing position (lower is better)
test_df = test_df.sort_values('PredictedPosition')

print("\nPredicted Japanese GP (Suzuka) 2025 Results:")
print(test_df[['Driver', 'Team', 'GridPosition', 'Experience', 'RainProbability', 
               'WeatherCondition', 'WetPerformanceFactor', 'PredictedPosition']])

# Visualize predictions as a bar chart
plt.figure(figsize=(14, 10))
sns.set_style("whitegrid")
ax = sns.barplot(x='PredictedPosition', y='Driver', hue='Team', data=test_df.sort_values('PredictedPosition'))
plt.title('🏎️ 2025 Japanese GP (Suzuka) Prediction: Expected Finishing Position', fontsize=16)
plt.xlabel('Predicted Position (lower is better)', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.legend(title='Team', loc='center right')
plt.tight_layout()
plt.savefig('japanese_gp_prediction.png', dpi=300)
plt.close()

print("\nPrediction visualizations saved as PNG files")
print("\nPrediction complete! Note that this is a simplified model; actual results may vary.")
print("Race Day: April 6, 2025 at 14:00 in Suzuka, Japan.")


# In[ ]:




