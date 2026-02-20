from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WineInput(BaseModel):
    fixed_acidity: float
    alcohol: float

@app.post("/predict")
def predict(data: WineInput):

    # Fake calculation for testing
    score = data.fixed_acidity * 0.5 + data.alcohol * 1.5

    return {
        "calculated_score": round(score, 2)
    }
