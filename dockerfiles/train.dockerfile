# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements_pipreqs.txt requirements.txt
# COPY data/fruits_vegetables_dataset/processed_data data/fruits_vegetables_dataset/processed_data
COPY README.md README.md
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/mlops_grp5/train.py"]