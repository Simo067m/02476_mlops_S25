import os
import requests
import streamlit as st
import pandas as pd
from PIL import Image
from google.cloud import run_v2

@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    # Update <project> and <region> with your GCP details
    parent = "projects/premium-portal-447810-a6/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        # Update `onnx-api` to match your backend service name
        if service.name.split("/")[-1] == "mlops-onnx-api":
            return service.uri
    # Fallback to an environment variable
    return os.environ.get("ONNX_BACKEND_URL", None)

def classify_image(image, backend_url):
    """Send the image to the ONNX backend for classification."""
    predict_url = f"{backend_url}/predict/"

    # Reset the file pointer of the uploaded file to the beginning
    image.seek(0)

    # Send file with correct MIME type
    files = {"file": (image.name, image, "image/jpeg")}
    response = requests.post(predict_url, files=files)

    # Debug log the response
    # print(response.json())  # Log the response for debugging

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        return None


def main():
    """Main function for the Streamlit app."""
    # Fetch the backend URL
    backend_url = get_backend_url()
    if not backend_url:
        st.error("Backend service URL not found. Please check your GCP settings or environment variables.")
        return

    st.title("ONNX Image Classification")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Send the image to the backend for classification
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                result = classify_image(uploaded_file, backend_url)
                if result:
                    # Display the result
                    st.subheader(f"Prediction: {result['predicted_class']}")
                    st.write("Confidence Scores:")

                    # Prepare data for bar chart
                    labels = ["Fresh", "Rotten"]
                    probabilities = result.get("confidence_scores", [0, 0])

                    # Debug log the probabilities
                    # print(f"Confidence Scores: {probabilities}")

                    # Check if probabilities are valid
                    if len(probabilities) == 2 and all(isinstance(p, (int, float)) for p in probabilities):
                        data = pd.DataFrame({"Class": labels, "Confidence": probabilities})
                        st.bar_chart(data.set_index("Class"))
                    else:
                        st.error("Invalid confidence scores received from the backend.")

if __name__ == "__main__":
    main()
