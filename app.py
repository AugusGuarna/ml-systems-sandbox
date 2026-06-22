from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
import joblib
import numpy as np
from uuid import uuid4
import structlog
import time
from pydantic_settings import BaseSettings, SettingsConfigDict

# This are pydantic classes which ensure that the input and output of the API are in the correct format.
class PredictRequest(BaseModel):
    features: list[float] = Field(...,min_length = 4, max_length = 4)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    request_id: str

class Settings(BaseSettings):
    # We need this in order to read from the .env
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    # Env variables
    model_path: str
    api_version: str

# This reshape is done as SKLearn models expect 2D array as input, even for single prediction
def correct_shape(features: list[float]):
    return np.array(features).reshape(1, -1)

settings = Settings()

app = FastAPI()

model = joblib.load(settings.model_path)

# We use structlog for logging, it is a structured logging library that allows us to log in a structured way, which is useful for debugging and monitoring.
log = structlog.get_logger()

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def prediction(request: PredictRequest):
    start_time = time.perf_counter()
    # As a prediction is the argmax over the probabilities we compute it once so we do not have to call the model twice which can be heavy
    # Think that every call to the model is a call to a heavy deep learning model that has to be lifted, we do not want to call it twice for the same input
    probabilities = model.predict_proba(correct_shape(request.features))[0]
    # End the timer
    end_time = time.perf_counter()
    latency = (end_time - start_time)*1000
    predicted_value = np.argmax(probabilities)
    probability = probabilities.max()
    request_id = uuid4()
    # Log information in structlog logger must start with event
    log.info(event="prediction",request_id = str(request_id), input = request.features, output = int(predicted_value), model_version = settings.api_version, latency_ms = latency)
    return {"prediction": int(predicted_value), "probability": float(probability),
            "model_version": settings.api_version, "request_id": str(request_id)}