from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
import joblib
import numpy as np
from uuid import uuid4

# This are pydantic classes which ensure that the input and output of the API are in the correct format.
class PredictRequest(BaseModel):
    features: list[float] = Field(...,min_length = 4, max_length = 4)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    request_id: str

app = FastAPI()

model = joblib.load('models/v1.pkl')

# This reshape is done as SKLearn models expect 2D array as input, even for single prediction
def correct_shape(features: list[float]):
    return np.array(features).reshape(1, -1)

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def prediction(request: PredictRequest):
    # As a prediction is the argmax over the probabilities we compute it once so we do not have to call the model twice which can be heavy
    # Think that every call to the model is a call to a heavy deep learning model that has to be lifted, we do not want to call it twice for the same input
    probabilities = model.predict_proba(correct_shape(request.features))[0]
    predicted_value = np.argmax(probabilities)
    return {"prediction": int(predicted_value[0]), "probability": float(probabilities.max()),
            "model_version": "v1", "request_id": str(uuid4())}