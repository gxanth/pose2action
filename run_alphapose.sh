#!/bin/bash
# Usage: ./run_alphapose.sh input.mp4 output_dir

# Initialize environment variables
IMAGE_NAME=alphapose
DOCKERFILE=docker/alphapose/Dockerfile
# Initialize mount points
DATA_DIR="$(pwd)/data/raw_videos"
OUTPUTS_DIR="$(pwd)/outputs"
BASE_DIR_IN_CONTAINER="/workspace"
YOLO_WEIGHTS_DIR="$(pwd)/external/alphapose/detector/yolo/data"
PRETRAINED_MODELS_DIR="$(pwd)/external/alphapose/pretrained_models"


# Check if the image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "Docker image '$IMAGE_NAME' not found. Building it now..."
  docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

# Run AlphaPose in the container
# Mount alphapose code, data, models, and output directories
docker run --gpus all -it --rm \
  -v "${DATA_DIR}":"${BASE_DIR_IN_CONTAINER}/data/raw_videos" \
  -v "${OUTPUTS_DIR}":"${BASE_DIR_IN_CONTAINER}/outputs" \
  -v "${YOLO_WEIGHTS_DIR}":"/workspace/alphapose/alphapose/detector/yolo/data" \
  -v "${PRETRAINED_MODELS_DIR}":"/workspace/alphapose/alphapose/pretrained_models" \
  "$IMAGE_NAME" bash

