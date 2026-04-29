import os
import pandas as pd
from train_model import generate_synthetic_data, train_and_save_model

def test_generate_synthetic_data():
    df = generate_synthetic_data(num_rows=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    expected_cols = [
        "Timestamp", "Vibration_Velocity_mm_s", 
        "Motor_Temperature_C", "Acoustic_Frequency_Hz", 
        "Load_Tonnage", "Remaining_Useful_Life_Hours"
    ]
    for col in expected_cols:
        assert col in df.columns

def test_train_and_save_model(tmp_path):
    df = generate_synthetic_data(num_rows=200)
    model_path = tmp_path / "test_model.pkl"
    train_and_save_model(df, model_path=str(model_path))
    assert os.path.exists(model_path)
