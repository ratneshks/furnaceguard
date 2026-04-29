# FurnaceGuard: AI Predictive Maintenance

**Tandem Cold Rolling Mill — Live Telemetry Dashboard**

An end-to-end AI-powered predictive maintenance system built for Tata Steel's Tandem Cold Rolling Mills. The application simulates real-time IoT sensor data, runs it through a trained Random Forest model to predict Remaining Useful Life (RUL), and displays results on a corporate-grade live dashboard.

## Features
- 🧠 **AI-Powered RUL Prediction** — Random Forest model trained on 10,000 synthetic sensor readings
- 📊 **Live Sensor Feeds** — Real-time charts for Vibration, Temperature, Acoustic Frequency & Load
- 🚨 **3-Tier Alert System** — Green (Nominal), Amber (Warning), Red (Critical) with animated indicators
- 💰 **ROI Cost Savings Tracker** — Estimated downtime savings in ₹ Lakhs
- 🔧 **Maintenance Recommendations** — Actionable next steps based on predicted RUL
- 📋 **Alert Action Log** — Timestamped record of all system alerts

## Tech Stack
- **Frontend:** Streamlit with custom CSS
- **Charting:** Plotly Graph Objects
- **ML:** scikit-learn (Random Forest Regressor)
- **Data:** Pandas, NumPy

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).
