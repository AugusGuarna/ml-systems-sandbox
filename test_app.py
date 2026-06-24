# FastAPI has a module called TestClient which lets us spin up our API in memory 
from fastapi.testclient import TestClient
from app import app # We need to import the API

client = TestClient(app) # We instantiate the testing client


def test_check_health(): #We define a function that does not receive any parameter
    response = client.get("/health") # We ask for the response from the API for that endpoint
    assert response.status_code == 200 # We need to test that the answer is correct
    assert response.json() == {"status":"ok"} # We check for the output

def test_check_prediction_w_correct_input():
    response = client.post("/predict",json={"features": [0, 0, 0, 0]}) # We need to pass the features
    assert response.status_code == 200 # We need to test that the answer is correct
    # It does not make sense to try and predict the decimals of the probability
    assert response.json()["prediction"] == 0 # This is a dict
    assert response.json()["probability"] >= 0
    assert response.json()["probability"] <= 1

def test_check_prediction_w_incorrect_input():
    response = client.post("/predict",json={"features": [0,0,0]}) # We need to pass the features
    assert response.status_code == 422 # We try to keep it as simple as possible

