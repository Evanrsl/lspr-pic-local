# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY app/requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY app/ /app/

# Make port 9090 available to the world outside this container
EXPOSE 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
# CMD ["python", "main.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:9090", "wsgi:app"]
