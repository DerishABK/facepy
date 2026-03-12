# Use a clean, modern Python base
FROM python:3.10-slim

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install dependencies one by one to avoid conflicts
# Use dlib-bin for binary install, and face-recognition with no-deps
RUN pip install --no-cache-dir dlib-bin
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir face-recognition --no-deps

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "face_attendance:app"]
