FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Create necessary runtime directories
RUN mkdir -p logs models

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
