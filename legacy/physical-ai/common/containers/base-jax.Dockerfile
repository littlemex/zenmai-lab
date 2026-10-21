# Base image for JAX-based experiments (π0 / openpi, MuJoCo MJX, Octo).
# JAX and PyTorch should NOT be mixed in the same container.

ARG CUDA_TAG=12.8.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      git curl ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
      "jax[cuda12]" flax optax

WORKDIR /workspace
