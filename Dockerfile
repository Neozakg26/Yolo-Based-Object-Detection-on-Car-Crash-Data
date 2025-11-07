# Use Python 3.11.9 base image
FROM python:3.11.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install YOLOv11 + dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir ultralytics opencv-python

# Default command (can be overridden at runtime)
CMD ["python3", "main.py"]
