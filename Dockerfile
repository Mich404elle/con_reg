'''
Uses Python 3.12 slim image as base
Sets up /app as working directory
Installs your requirements
Copies your code
Exposes port 8080
Defines how to run your app
'''

# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy all project files to container
COPY . .

# Make port 8080 available
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]