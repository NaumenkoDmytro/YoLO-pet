FROM python:3.10

WORKDIR /app

COPY ../requirements.txt /app/

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN set -xe \
    && pip install --upgrade pip

RUN pip install -r requirements.txt


# Download YOLO model SERVER(LINUX)
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
