FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY latent_zoom.py .
COPY image_base_512.png .

CMD ["python", "latent_zoom.py"]
