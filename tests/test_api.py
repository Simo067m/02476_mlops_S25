from fastapi.testclient import TestClient
from mlops_grp5.api import app  # Ensure this path is correct for your project structure
import os
import random

client = TestClient(app)

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "FastAPI Inference Application is running!"}

def test_valid_predict():
    with TestClient(app) as client:
        test_image_path = "tests/data/fresh_sample.jpg"
        with open(test_image_path, "rb") as f:
            response = client.post("/predict/", files={"file": ("fresh_sample.jpg", f, "image/jpeg")})
        assert response.status_code == 200
        assert "predicted_class" in response.json()
        assert response.json()["predicted_class"] in ["Fresh", "Rotten"]

def test_invalid_prediction():
    with TestClient(app) as client:
        response = client.post("/predict/", files={"file": ("text.txt", b"Invalid content", "text/plain")})
        assert response.status_code == 400  # Now expect a 400 Bad Request
        assert response.json() == {"detail": "Invalid image file uploaded"}