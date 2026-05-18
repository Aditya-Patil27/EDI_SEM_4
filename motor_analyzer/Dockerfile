# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing / scipy
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/>

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run the pipeline test when the container launches
CMD ["python", "pipeline.py", "--all"]
