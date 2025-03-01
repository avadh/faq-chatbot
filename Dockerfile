# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the necessary ports
EXPOSE 5000 7860

# Run Gradio + FastAPI together
CMD ["python", "src/frontend.py"]


