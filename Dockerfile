# Use a pre-built face_recognition image to avoid memory-intensive compilation
FROM animcogn/face_recognition:latest

# Install additional system dependencies for OpenCV and Flask
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the essentials first
COPY requirements.txt .

# Install additional Python dependencies (face-recognition will already be satisfied)
RUN pip install --no-cache-dir Flask Flask-Cors requests opencv-python-headless gunicorn

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "face_attendance:app"]
