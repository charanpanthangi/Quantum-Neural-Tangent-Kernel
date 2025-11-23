# Use slim Python image to keep container light.
FROM python:3.11-slim

# Install system build tools for scientific packages.
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set working directory.
WORKDIR /app

# Copy dependency list and install Python packages.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container.
COPY . .

# Default command runs the demo with sensible defaults.
CMD ["python", "app/main.py", "--n-points", "20", "--steps", "200"]
