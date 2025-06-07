FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY conda.yaml ./
RUN pip install --no-cache-dir \
    pandas>=1.3.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.0.0 \
    mlflow>=1.30.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0

# Copy all project files
COPY . .

# Create mlruns directory
RUN mkdir -p mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the model training
CMD ["python", "modelling.py"]
