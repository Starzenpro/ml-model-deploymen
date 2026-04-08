FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model_v1.joblib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
