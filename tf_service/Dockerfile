FROM python:3.11.7-slim-bookworm

# Set working directory
WORKDIR /app
COPY ./tf_service /app/tf_service
# Copy requirements and install
RUN pip install --no-cache-dir -r /app/tf_service/requirements.txt

# Copy the rest of the code
WORKDIR /app/tf_service

# Expose FastAPI port
EXPOSE 8000
# By default start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
