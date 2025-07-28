# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run the processing script
CMD ["python", "process_all_pdfs.py"]
