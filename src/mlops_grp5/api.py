from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from typing import Dict
from mlops_grp5.model import ImageModel
import torch
from torchvision import transforms
from fastapi import HTTPException

MODEL_CHECKPOINT_PATH = "models/model.pth"
model = None
CLASS_LABELS = {0: "Fresh", 1: "Rotten"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic for the FastAPI app."""
    global model
    model = ImageModel.load_trained_model()  # Initialize and load the model
    print("Model loaded successfully.")
    yield  # The app runs here
    print("Shutting down application...")

# Create the FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "FastAPI Inference Application is running!"}

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the input image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict:
    """Predict the class of the uploaded image."""
    try:
        image = Image.open(file.file).convert("RGB")  # Ensure the image is in RGB mode
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded")

    input_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = output.max(1)

    predicted_label = CLASS_LABELS[predicted_class.item()]

    return {"predicted_class": predicted_label}
