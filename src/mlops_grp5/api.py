from fastapi import FastAPI, UploadFile, File
from PIL import Image
from typing import Dict
from mlops_grp5.model import ImageModel
import torch
from torchvision import transforms

MODEL_CHECKPOINT_PATH = "models/model.pth"
model = None
CLASS_LABELS = {0: "Fresh", 1: "Rotten"}

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI Inference Application is running!"}

@app.on_event("startup")
async def load_model():
    global model
    model = ImageModel.load_trained_model()
    print("Model loaded successfully.")

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
    image = Image.open(file.file).convert("RGB")  # Ensure the image is in RGB mode
    input_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = output.max(1)

    predicted_label = CLASS_LABELS[predicted_class.item()]

    return {"predicted_class": predicted_label}
