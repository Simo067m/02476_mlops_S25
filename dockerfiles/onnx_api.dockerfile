FROM python:3.11-slim

WORKDIR /app

# Install dependencies for ONNX and Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements and install them
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install onnxruntime

# Copy only the relevant parts of the project
COPY src /app/src
COPY models /app/models

# Add the `src` directory to PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose the default Cloud Run port
EXPOSE $PORT

# Run the FastAPI ONNX application
CMD exec uvicorn src.mlops_grp5.onnx_api:app --port $PORT --host 0.0.0.0
