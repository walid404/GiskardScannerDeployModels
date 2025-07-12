from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import uvicorn
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# === FastAPI app === #
app = FastAPI()

# ========================= You Can Update From Here ========================= #


# === Load Model === #
model = joblib.load("regressor.pkl")

# === Features used in model === #
features_model = [
    'battery_capacity_kWh', 'efficiency_wh_per_km', 'drivetrain',
    'fast_charging_power_kw_dc', 'car_body_type', 'torque_nm',
    'acceleration_0_100_s', 'towing_capacity_kg', 'seats',
    'length_mm', 'width_mm', 'height_mm'
] # Ensure the order matches the model training process


# === Input Schema === #
class CarFeatures(BaseModel):
    battery_capacity_kWh: float
    efficiency_wh_per_km: float
    drivetrain: str
    fast_charging_power_kw_dc: float
    car_body_type: str
    torque_nm: float
    acceleration_0_100_s: float
    towing_capacity_kg: float
    seats: float
    length_mm: float
    width_mm: float
    height_mm: float

# === Preprocessing === #
def preprocess(df):
    # Ensure that all NaN values are properly handled prior to sending the CSV
    # to the scanner — this hasn’t been done here.

    label_cols = ['drivetrain', 'car_body_type']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# ================================ Until Here ================================ #


# === Endpoint: Predict Single === #
@app.post("/predict_point")
async def predict_one(data: CarFeatures):
    try:
        df = pd.DataFrame([data.dict()])
        df = df[features_model]
        df = preprocess(df)
        prediction = model.predict(df)
        return JSONResponse(content={"predictions": prediction[0]}, status_code=200) # Don't Change the predictions names in JSONResponse
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === Endpoint: Predict Batch === #
@app.post("/predict_batch")
async def predict_batch(data: List[CarFeatures]):
    try:
        df = pd.DataFrame([d.dict() for d in data])
        df = df[features_model]
        df = preprocess(df)
        predictions = model.predict(df)
        return JSONResponse(content={"predictions": predictions.tolist()}, status_code=200) # Don't Change the predictions names in JSONResponse
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === Run the server === #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
