"""
data_stream.py — Real-Time IoT Sensor Data Simulator
=====================================================
Simulates the output of edge sensors mounted on a Tandem Cold Rolling Mill.
The generator alternates between healthy operation and degrading states,
mimicking real-world bearing wear, motor overheating, and acoustic anomalies.

NOTE: time.sleep is NOT called here — the caller (app.py) controls tick rate.
"""

import random
from datetime import datetime


def get_realtime_data_generator():
    """
    Generator yielding one sensor reading dict per call.
    Alternates between 'healthy' and 'degrading' states to let the
    AI model demonstrate its prediction capability in real-time.
    """
    state = "healthy"
    state_duration = random.randint(15, 25)  # seconds in this state
    step = 0
    degradation_factor = 0.0

    # Baseline sensor values for a healthy Tandem Cold Rolling Mill
    BASE_VIBRATION = 2.5   # mm/s — ISO 10816 zone A for heavy machinery
    BASE_TEMP = 58.0       # °C — normal motor operating temperature
    BASE_ACOUSTIC = 1000.0 # Hz — baseline acoustic emission frequency
    BASE_LOAD = 30.0       # Tonnes — nominal rolling load

    while True:
        # Switch state when duration expires
        if step >= state_duration:
            if state == "healthy":
                state = "degrading"
                state_duration = random.randint(20, 40)  # longer degradation window
            else:
                state = "healthy"
                state_duration = random.randint(15, 25)
            step = 0
            degradation_factor = 0.0

        # Degradation ramps up non-linearly (simulates accelerating wear)
        if state == "degrading":
            degradation_factor += 0.035
        wear = degradation_factor ** 1.8 if state == "degrading" else 0.0

        # Simulating bearing wear-and-tear via vibration spikes
        vibration = BASE_VIBRATION + (wear * 6.0) + random.gauss(0, 0.3)
        # Motor temperature rises as friction increases in degraded bearings
        temp = BASE_TEMP + (wear * 35.0) + random.gauss(0, 1.5)
        # Acoustic frequency shifts upward as micro-cracks develop
        acoustic = BASE_ACOUSTIC + (wear * 350.0) + random.gauss(0, 25.0)
        # Load fluctuates based on strip thickness & rolling schedule
        load = BASE_LOAD + random.uniform(-8.0, 12.0) + (wear * 3.0)

        reading = {
            "Timestamp": datetime.now(),
            "Vibration_Velocity_mm_s": max(0.5, round(vibration, 3)),
            "Motor_Temperature_C": max(25.0, round(temp, 2)),
            "Acoustic_Frequency_Hz": max(200.0, round(acoustic, 1)),
            "Load_Tonnage": max(10.0, round(load, 2)),
            "State": state,
        }

        yield reading
        step += 1
        # NOTE: No time.sleep() here — the dashboard loop controls timing
