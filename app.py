from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WineFeatures(BaseModel):
    fixedAcidity: float
    volatileAcidity: float
    citricAcid: float
    residualSugar: float
    chlorides: float
    freeSulfurDioxide: float
    totalSulfurDioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict_wine(data: WineFeatures):

    input_data = np.array([[
        data.fixedAcidity,
        data.volatileAcidity,
        data.citricAcid,
        data.residualSugar,
        data.chlorides,
        data.freeSulfurDioxide,
        data.totalSulfurDioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    return {
        "quality": int(prediction[0])
    }
