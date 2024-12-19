FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install project
RUN pip install --no-cache-dir .

RUN pip install jinja2 pyyaml

# (Optional) Download a pretrained model and its config if you plan to use one for fine-tuning or inference
# ARG PRETRAINED_MODEL_NAME="stabilityai/stable-audio-open-1.0"
# RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('$PRETRAINED_MODEL_NAME', filename='model_config.json', repo_type='model'); hf_hub_download('$PRETRAINED_MODEL_NAME', filename='model.safetensors', repo_type='model')"

# Create a directory for saving checkpoints
RUN mkdir /app/checkpoints

# Define default command (you'll likely override this when running the container)
CMD ["python", "train.py", "--model-config", "/app/model_config.json", "--dataset-config", "/app/dataset_config.json", "--save-dir", "/app/checkpoints"]
