# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the prompthelix application directory into the container at /app
COPY ./prompthelix ./prompthelix

# Copy the main application file if it's in the root, otherwise adjust path
# Assuming main.py is in the prompthelix directory based on previous steps
COPY ./prompthelix/main.py ./prompthelix/main.py

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
#ENV MODULE_NAME prompthelix.main
#ENV VARIABLE_NAME app
ENV MODULE_NAME=prompthelix.main
ENV VARIABLE_NAME=app


# Run app.py when the container launches
# Ensure the command correctly points to the FastAPI app instance in prompthelix/main.py
CMD ["uvicorn", "prompthelix.main:app", "--host", "0.0.0.0", "--port", "8000"]
