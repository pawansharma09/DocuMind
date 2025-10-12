# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file into the container at /app
COPY main.py .

# Expose the port the app runs on
# Render will automatically set the PORT environment variable.
EXPOSE 8000

# Define the command to run the application
# We use the $PORT variable provided by Render.
# The --host 0.0.0.0 makes the app accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
