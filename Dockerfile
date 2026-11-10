# Entorno reproducible: Python + MPICH + dependencias.
#   docker build -t kmeans-paralelo .
#   docker run --rm -v "$PWD/resultados:/app/resultados" kmeans-paralelo
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        mpich libmpich-dev build-essential util-linux pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

# Por defecto: verifica correctitud y corre toda la bateria de experimentos.
CMD ["sh", "-c", "python experimentos/verificar_correctitud.py && python experimentos/ejecutar_experimentos.py"]
