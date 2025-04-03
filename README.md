# F1 Japanese Grand Prix 2025 Prediction Model

A **machine-learning–based prediction system** for forecasting Formula 1 race results for the **2025 Japanese Grand Prix at Suzuka**, using **historical F1 data**, **wet-weather performance** analysis, and **OpenWeatherMap** forecasts to account for rainy conditions.

## Project Overview

This project leverages **historical Formula 1 data** from the **2022–2024** seasons to build a **predictive model** that forecasts finishing positions for the **2025 Japanese Grand Prix**. The model incorporates:

- **Historical driver performance**  
- **Wet-driver analysis** (Canadian GP dry vs. rain)  
- **Weather data** (rain probability, temperature, humidity)  
- **Circuit-specific performance patterns**  
- **Team changes for 2025** (e.g., Hamilton moving to Ferrari)  
- **Rookie driver integration** (via fallback or synthetic data)

By combining **wet-weather metrics** and **real-time forecasts** for a potentially **rainy** Suzuka GP, this system produces more **accurate** predictions than a standard dry-weather model.

## Key Features

1. **Data Collection**  
   - Automated fetching of **historical F1 race data** (2022–2024) using **FastF1**  
   - **Team reassignments** and **rookie additions** accounted for with fallback data  
   - Driver **experience** factors and **sample weighting** for recent races

2. **Wet-Driver Analysis**  
   - **Canadian GP Comparison**: Evaluates driver performance under **dry (2022)** vs. **wet (2023)** conditions  
   - Computes a **Wet Driver Score** to reflect each driver’s **rain performance** advantage or disadvantage

3. **Weather Integration**  
   - **OpenWeatherMap** API call for **rain probability**, **temperature**, and **humidity** at Suzuka on **April 6, 2025 (14:00)**  
   - Dynamically adjusts predictions if **rain_probability** is above a threshold

4. **Random Forest Model**  
   - **Weighted sample data** for more recent races (2024 > 2023 > 2022)  
   - Incorporates **Wet Driver Score** and **weather features** to refine finishing position predictions  
   - **Fallback logic** when certain data is incomplete (e.g., synthetic race results)

## Visualizations

1. **Grid Position vs Predicted Finish**  
   **File:** `grid_vs_finish.png`  
   - Plots each driver’s **starting grid** against **expected finishing position**  
   - Diagonal line indicates perfect parity between start and finish; deviations highlight **wet-performance** influence

2. **Driver Performance Ranking**  
   **File:** `japanese_gp_prediction.png`  
   - Bar chart displaying **predicted finishing positions** for each driver  
   - **Color-coded** by team; **lower bars** indicate stronger performance  
   - Reflects **rain adjustments** and **wet driver scores**

## Results

The model predicts a podium of:

1. 🥇 **Charles Leclerc** (Ferrari)  
2. 🥈 **Max Verstappen** (Red Bull Racing)  
3. 🥉 **Carlos Sainz Jr.** (Williams)

The full prediction includes expected finishing positions for all 20 drivers competing in the **2025 Japanese Grand Prix (Suzuka)**, factoring in:

- ✅ **Real-time weather forecast** retrieved via [OpenWeatherMap API](https://openweathermap.org/api)  
- ✅ **Wet Driver Scores** calculated from comparative analysis of **2022 (dry)** and **2023 (wet)** Canadian GPs using [FastF1](https://theoehrly.github.io/Fast-F1/)  
- ✅ **Driver experience**, **team changes**, and **team performance metrics**  
- ✅ **Machine learning model** trained with weighted historical data (2022–2024)

---

### References

- 🔗 [FastF1 - Formula 1 Telemetry & Data API](https://theoehrly.github.io/Fast-F1/)  
- 🔗 [OpenWeatherMap - Weather API for Real-Time Forecasts](https://openweathermap.org/api)  
- 🔗 [Formula 1 Official Website](https://www.formula1.com/)  
- 🔗 [Scikit-learn Documentation](https://scikit-learn.org/stable/)  

---

With **wet-weather metrics**, **team updates**, and **real-time weather data**, this model delivers a more **realistic** forecast for the **2025 Japanese Grand Prix**. Feel free to explore or enhance the project by adding new features, experimenting with other ML algorithms, or extending the weather forecast horizon!  
