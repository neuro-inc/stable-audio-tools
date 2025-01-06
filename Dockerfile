FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install the project
RUN pip install --no-cache-dir -e .

# Use a newer version of NumPy that includes np.dtypes
RUN pip install --no-cache-dir numpy==1.25.2

RUN mkdir -p /app/checkpoints

CMD ["python", "train.py", "--model-config", "/app/model_config.json", "--dataset-config", "/app/dataset_config.json", "--name", "my_model_name", "--save-dir", "/app/checkpoints", "--checkpoint-every", "100", "--batch-size", "32", "--num-gpus", "1", "--precision", "16-mixed", "--seed", "42", "--pretrained-ckpt-path", "./checkpoints/model-001.ckpt"]
