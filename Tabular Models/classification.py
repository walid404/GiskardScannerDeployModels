from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import uvicorn
import warnings

warnings.filterwarnings("ignore")


# === FastAPI app === #
app = FastAPI()

# ========================= You Can Update From Here ========================= #

# === Load trained model and training features === #
with open("classifier.pkl", "rb") as f:
    model = pickle.load(f)


# === Constants for binning === #
AGE_BINS = [0, 19, 29, 39, 49, 59, 69, 80]
AGE_CATEGORIES = ["<20s", "20s", "30s", "40s", "50s", "60s", ">60s"]

NA_TO_K_BINS = [0, 9, 19, 29, 50]
NA_TO_K_CATEGORIES = ["<10", "10-20", "20-30", ">30"]

features_model = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"] # Ensure the order matches the model training process

# === Input schema === #
class InputData(BaseModel):
    Age: float
    Sex: str
    BP: str
    Cholesterol: str
    Na_to_K: float

# === Preprocessing functions === #
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Perform binning for Age and Na_to_K."""
    # Ensure that all NaN values are properly handled prior to sending the CSV
    # to the scanner — this hasn’t been done here.

    df = df.copy()
    df['Age'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_CATEGORIES, right=False, include_lowest=True)
    df['Na_to_K'] = pd.cut(df['Na_to_K'], bins=NA_TO_K_BINS, labels=NA_TO_K_CATEGORIES, right=False, include_lowest=True)
    return df

# ================================ Until Here ================================ #


# === Prediction endpoint === #
@app.post("/predict_point")
async def predict(data: InputData):
    try:
        # Step 1: Create DataFrame
        df = pd.DataFrame([data.dict()])
        df = df[features_model]

        # Step 2: preprocess
        df = preprocessing(df)

        # Step 3: Predict
        prediction = model.predict_proba(df) # Must return the probabilities

        return JSONResponse(content={"predictions": prediction[0].tolist()}, status_code=200) # Don't Change the predictions names in JSONResponse

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# === Batch Prediction endpoint === #
@app.post("/predict_batch")
async def predict(data: List[InputData]):
    try:
        # Convert list of InputData into DataFrame
        df = pd.DataFrame([item.dict() for item in data])

        # Preprocess
        df = df[features_model]
        df = preprocessing(df)


        # Predict
        predictions = model.predict_proba(df) # Must return the probabilities

        # Convert to list of lists
        return JSONResponse(content={"predictions": predictions.tolist()}, status_code=200) # Don't Change the predictions names in JSONResponse

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)