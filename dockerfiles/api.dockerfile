FROM python:3.11-slim

WORKDIR /app

# Install dependencies for building Python packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements and install them
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy only the relevant parts of the project
COPY src /app/src
COPY models /app/models

# Add the `src` directory to PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose the default Cloud Run port
EXPOSE $PORT

# Run the FastAPI application
CMD exec uvicorn src.mlops_grp5.api:app --port $PORT --host 0.0.0.0
