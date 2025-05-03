
# 🏎️ F1 Japanese Grand Prix 2025 – Race Prediction Model

> A machine-learning–based prediction system for forecasting the **2025 Japanese GP at Suzuka**, combining historical race data, rain-performance metrics, and real-time weather forecasting.

---

## 📌 Project Overview

This project leverages data from the **2022–2024 F1 seasons** to predict driver performance for the 2025 Japanese GP at Suzuka.

It accounts for:
- 🧠 Historical performance & wet-weather skill
- 🌧️ OpenWeatherMap forecast for April 6, 2025 (14:00)
- 🏎️ Circuit-specific behavior and team changes (e.g., Hamilton to Ferrari)

---

## 📊 Predicted Driver Performance

![Predicted Finishing Positions](./assets/japanese_gp_prediction.png)

This chart compares predicted finishing positions across all 20 drivers, color-coded by team.

---

## 🧠 How the Model Works

```mermaid
graph TD
  A[Historical F1 Data 2022-2024] --> B[Feature Engineering]
  B --> C[Wet Driver Score (Rain vs Dry Analysis)]
  B --> D[Weather Forecast (OWM API)]
  C --> E[Random Forest Model]
  D --> E
  E --> F[Final Grid Predictions]
```

---

## 🌧️ Rain Integration Logic

- Real-time weather pulled from OpenWeatherMap for Suzuka on race day
- Adjusts prediction weight if rain probability > 40%
- Computes **Wet Driver Score** from 2022 (dry) vs. 2023 (wet) Canadian GP comparison

---

## 🥇 Predicted Podium

| Position | Driver              | Team              |
|----------|---------------------|-------------------|
| 🥇 1st   | Charles Leclerc     | Ferrari           |
| 🥈 2nd   | Max Verstappen      | Red Bull Racing   |
| 🥉 3rd   | Carlos Sainz Jr.    | Williams          |

---

## 🧩 Key Features

- **FastF1**: historical telemetry + timing
- **OpenWeatherMap**: real-time forecast
- **Random Forest**: trained with weighted historical performance
- **Fallback Logic**: for rookies and incomplete driver data

---

## 🔍 Next Improvements

- Add confidence intervals to predictions
- Extend to Qualifying + Fastest Lap predictions
- Explore neural nets or LSTM-based dynamic modeling

---

## 📚 References

- 🔗 [FastF1 API](https://theoehrly.github.io/Fast-F1/)
- 🔗 [OpenWeatherMap](https://openweathermap.org/api)
- 🔗 [Formula 1 Official Site](https://www.formula1.com)
- 🔗 [Scikit-learn Documentation](https://scikit-learn.org)

---

## 👨‍💻 Author

**Manan Upadhyay**  
📫 [Connect on LinkedIn](https://www.linkedin.com/in/mananupadhyay2000/)

---

