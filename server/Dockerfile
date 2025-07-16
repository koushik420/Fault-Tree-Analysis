# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set environment variable to disable buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
COPY .env /code/.env

# Expose port 5000 for Flask
EXPOSE 80

# Run the app using Gunicorn with multiple workers & threads
CMD ["gunicorn", "-b", "0.0.0.0:80", "--workers", "2", "--threads", "4", "app:app"]
