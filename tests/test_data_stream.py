import types
from data_stream import get_realtime_data_generator

def test_get_realtime_data_generator():
    gen = get_realtime_data_generator()
    assert isinstance(gen, types.GeneratorType)
    
    reading1 = next(gen)
    assert isinstance(reading1, dict)
    assert "Vibration_Velocity_mm_s" in reading1
    assert "Motor_Temperature_C" in reading1
    assert "Timestamp" in reading1
    
    reading2 = next(gen)
    assert reading1["Timestamp"] != reading2["Timestamp"]
