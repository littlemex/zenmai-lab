# Base image for PyTorch-based experiments (Isaac Lab, GR00T, LeRobot, OpenVLA).
# Override CUDA_TAG / TORCH_VER as needed.

ARG CUDA_TAG=12.8.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ARG TORCH_VER=2.7.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      git curl ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
      torch==${TORCH_VER} torchvision \
      --index-url ${TORCH_INDEX}

WORKDIR /workspace
