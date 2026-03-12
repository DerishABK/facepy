# Use a clean, modern Python base
FROM python:3.10-slim-bullseye

# Install essential system dependencies for OpenCV and Face Recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install dependencies using the pre-compiled dlib-bin
# This avoids the memory-heavy compilation of dlib
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "face_attendance:app"]
