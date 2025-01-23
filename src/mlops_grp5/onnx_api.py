from contextlib import asynccontextmanager
from typing import Dict
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
import torch
from mlops_grp5.model import ImageModel
import os

MODEL_PATH = "models/model.onnx"

# Check if ONNX model already exists
if os.path.exists(MODEL_PATH):
    print(f"ONNX model already exists at {MODEL_PATH}. Skipping export.")
else:
    print("ONNX model not found. Exporting PyTorch model to ONNX...")

    # Load your trained PyTorch model
    model = ImageModel.load_trained_model()
    model.eval()

    # Dummy input to match the model's input shape
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=11  # Specify the ONNX opset version
    )
    print(f"ONNX model successfully exported to {MODEL_PATH}.")

onnx_session = None
CLASS_LABELS = {0: "Fresh", 1: "Rotten"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic for the FastAPI app."""
    global onnx_session
    onnx_session = ort.InferenceSession(MODEL_PATH)
    print("ONNX model loaded successfully.")
    yield  # The app runs here
    print("Shutting down ONNX API...")

# Create the FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "FastAPI ONNX Inference Application is running!"}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the input image for ONNX inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.numpy()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict:
    """Predict the class of the uploaded image using ONNX."""
    try:
        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")  # Ensure the image is in RGB mode
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded")

    input_tensor = preprocess_image(image)

    # Perform inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    predictions = onnx_session.run([output_name], {input_name: input_tensor})

    # Get probabilities (apply softmax)
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions[0]), dim=1).numpy()

    # Get predicted class
    predicted_class = np.argmax(probabilities)

    # Prepare the response
    response = {
        "predicted_class": CLASS_LABELS[predicted_class],
        "confidence_scores": probabilities[0].tolist(),  # Convert to list for JSON serialization
    }
    return response

