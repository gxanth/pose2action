# 
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

WORKDIR /workspace/alphapose

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    pkg-config \
    cython \
    && rm -rf /var/lib/apt/lists/*

COPY external/alphapose /workspace/alphapose


RUN pip install --upgrade pip setuptools wheel \
    && pip install cython \
    && pip install 'numpy<2'

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"
RUN pip install -e .

# Debug: list files in scripts directory
RUN ls -l /workspace/alphapose/scripts

CMD ["python", "scripts/demo_inference.py", "--help"]