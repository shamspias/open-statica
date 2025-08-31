FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-ml.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install ML dependencies
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
    pip install --no-cache-dir -r requirements-ml.txt; \
    fi

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p uploads models plugins

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]