"""
train_model.py — Synthetic Data Generation & ML Model Training
==============================================================
Generates 10,000 rows of realistic machinery sensor data with a physics-based
degradation formula simulating multiple failure cycles of a Tandem Cold Rolling Mill.
Trains a Random Forest Regressor to predict Remaining Useful Life (RUL).
Saves the model, scaler, and feature importance as a single .pkl artifact.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def generate_synthetic_data(num_rows=10000):
    """
    Generate synthetic machinery sensor data with a multi-cycle degradation formula.

    Physics rationale:
    - Vibration increases as bearings wear (ISO 10816 standard)
    - Motor temperature rises with increased friction
    - Acoustic emissions shift as micro-cracks form in roll surfaces
    - Load varies with strip thickness and rolling schedule
    - RUL decreases linearly within each degradation cycle
    """
    np.random.seed(42)
    start_time = datetime.now() - timedelta(days=365)

    # Baseline operating parameters (healthy machine)
    BASE_VIB = 2.5    # mm/s
    BASE_TEMP = 58.0   # °C
    BASE_ACOUSTIC = 1000.0  # Hz

    data = []
    current_time = start_time
    cycle_length = np.random.randint(400, 1200)
    step = 0

    for _ in range(num_rows):
        # Reset cycle when machine is "repaired" after failure
        if step >= cycle_length:
            cycle_length = np.random.randint(400, 1200)
            step = 0

        # Normalized degradation (0 = fresh, 1 = failure)
        degradation = step / cycle_length
        # Exponential wear curve — accelerates near end-of-life
        wear = degradation ** 2.5

        # Sensor simulations with correlated noise
        noise_factor = 1.0 + degradation * 0.5  # noise increases with wear
        vibration = BASE_VIB + (wear * 12.0) + np.random.normal(0, 0.4 * noise_factor)
        temp = BASE_TEMP + (wear * 45.0) + np.random.normal(0, 1.5 * noise_factor)
        acoustic = BASE_ACOUSTIC + (wear * 600.0) + np.random.normal(0, 40.0 * noise_factor)
        load = np.random.uniform(15.0, 48.0) + wear * 5.0  # load slightly increases under stress

        # RUL in hours: linearly decreases within each cycle
        rul_hours = max(0.0, (cycle_length - step) * 0.4)

        data.append({
            "Timestamp": current_time,
            "Vibration_Velocity_mm_s": max(0.5, vibration),
            "Motor_Temperature_C": max(25.0, temp),
            "Acoustic_Frequency_Hz": max(200.0, acoustic),
            "Load_Tonnage": max(10.0, load),
            "Remaining_Useful_Life_Hours": rul_hours,
        })

        step += 1
        current_time += timedelta(minutes=30)

    return pd.DataFrame(data)


def train_and_save_model(df, model_path="model.pkl"):
    """
    Trains a Random Forest Regressor on the synthetic data.
    Evaluates on a holdout set and saves model + scaler + feature importance.
    """
    print("=" * 60)
    print("  FurnaceGuard — ML Model Training Pipeline")
    print("=" * 60)

    features = [
        "Vibration_Velocity_mm_s",
        "Motor_Temperature_C",
        "Acoustic_Frequency_Hz",
        "Load_Tonnage",
    ]
    target = "Remaining_Useful_Life_Hours"

    X = df[features]
    y = df[target]

    # Train/test split for validation metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest with tuned hyperparameters
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE:  {mae:.2f} hours")
    print(f"  R²:   {r2:.4f}")

    # Feature importance for the dashboard
    importance = dict(zip(features, model.feature_importances_))
    print(f"  Feature Importances: {importance}")

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "metrics": {"mae": mae, "r2": r2},
        "feature_importance": importance,
    }
    joblib.dump(artifact, model_path)
    print(f"  Artifact saved → {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    print("Generating synthetic dataset (10,000 rows)...")
    df = generate_synthetic_data(10000)
    print(f"Dataset shape: {df.shape}")
    print(f"RUL range: {df['Remaining_Useful_Life_Hours'].min():.1f} — {df['Remaining_Useful_Life_Hours'].max():.1f} hours")
    train_and_save_model(df, "model.pkl")
    print("Done!")
