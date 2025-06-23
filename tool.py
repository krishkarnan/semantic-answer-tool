from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Vehicle Telematics API",
    version="1.0.0",
    description="API to fetch vehicle telematics data based on VIN"
)

class TelematicsResponse(BaseModel):
    vin: str
    timestamp: str
    engine_status: str
    engine_temperature_celsius: float
    rpm: int
    vehicle_speed_kph: float
    battery_voltage: float
    fuel_level_percent: float
    dtc_codes: list[str]
    location: Dict[str, float]
    odometer_km: float
    last_service_km: float
    coolant_level_percent: float
    oil_pressure_psi: float
    throttle_position_percent: float
    intake_air_temp_celsius: float

@app.get("/get-vehicle-telematics", response_model=TelematicsResponse)
def get_vehicle_telematics(vin_number: str = Query(..., description="Vehicle Identification Number")):
    return {
        "vin": vin_number,
        "timestamp": "2025-05-05T14:23:45Z",
        "engine_status": "on",
        "engine_temperature_celsius": 104.2,
        "rpm": 3200,
        "vehicle_speed_kph": 62,
        "battery_voltage": 12.3,
        "fuel_level_percent": 47.5,
        "dtc_codes": ["P0301", "P0171"],
        "location": {"lat": 37.7749, "lon": -122.4194},
        "odometer_km": 61240,
        "last_service_km": 45000,
        "coolant_level_percent": 78,
        "oil_pressure_psi": 28,
        "throttle_position_percent": 18.5,
        "intake_air_temp_celsius": 38.0
    }
