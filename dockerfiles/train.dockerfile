# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY data data/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir --verbose --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/mlops_grp5/train.py"]