FROM python:3.11.7-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

ENTRYPOINT ["gunicorn", "main:app", "-b", ":8001", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "60"]