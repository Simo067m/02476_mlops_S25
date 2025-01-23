FROM python:3.11-slim

# Install necessary dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements_frontend.txt /app/requirements_frontend.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

# Copy the application files (entire src folder)
COPY src /app/src

# Expose the default port for Cloud Run
EXPOSE 8080

# Use a startup script to resolve $PORT before running Streamlit
CMD ["sh", "-c", "streamlit run src/mlops_grp5/frontend.py --server.port=$PORT --server.address=0.0.0.0"]
