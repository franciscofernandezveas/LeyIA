FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias primero (mejor cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Railway asigna el puerto en la variable PORT
ENV PORT=8501
EXPOSE 8501

# Ejecuta Streamlit escuchando en 0.0.0.0:$PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
