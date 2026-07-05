FROM python:3.11-slim

# Dependencias del sistema para OpenCV y librerías gráficas
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PyTorch CPU primero (más liviano, evita descargar CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copiar e instalar el resto de dependencias
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copiar el código y el modelo
COPY sismoai/ sismoai/
COPY models/yolo_fallas.pt models/yolo_fallas.pt

# Streamlit no necesita secrets.toml en producción — usaremos variables de entorno
# (NO copies secrets.toml aquí)

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "sismoai/app_yolo.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.maxUploadSize=200"]
