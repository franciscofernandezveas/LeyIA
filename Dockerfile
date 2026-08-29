FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema para evitar errores de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el proyecto
COPY . .

# Puerto dinámico asignado por Railway
ENV PORT=8000
EXPOSE 8000

# Inicia la API FastAPI
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
