# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PyTorch, image processing, and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libstdc++6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to the container
COPY requirements.txt .

# Install Python dependencies
# Using pip cache for faster rebuilds if requirements haven't changed
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the container
COPY . .

# Set environment variables for Python and Flask
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

