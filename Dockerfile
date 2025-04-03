FROM python:3.11.7-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy the application code into the image
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Define the entrypoint
ENTRYPOINT ["gunicorn", "main:app", "-b", ":8000", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "60"]