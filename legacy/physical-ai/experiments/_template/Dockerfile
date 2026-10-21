# Replace with the right base for your experiment.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

CMD ["bash", "train.sh"]
