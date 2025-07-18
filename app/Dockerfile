FROM python:3.10

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    torchvision \
    pillow \
    python-dotenv \
    ultralytics \
    python-multipart \
    requests \
    psutil

# Evitar os.fork, forçar spawn
ENV PYTORCH_MULTIPROCESSING_START_METHOD=spawn
ENV PYTORCH_DISABLE_MKLDNN=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
