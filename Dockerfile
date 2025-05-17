FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

# Create necessary directories if they don't exist
RUN mkdir -p input output model_artifacts

# Copy data files specifically (before copying everything else)
# This ensures these directories are copied even if they're in .dockerignore
COPY input/ /app/input/
COPY output/ /app/output/

# Copy the rest of the application
COPY . .

# Expose the port your API will run on
EXPOSE 8080

# Command to run the API
CMD ["python", "resume_scanner_api.py"]