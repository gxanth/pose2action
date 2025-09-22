#!/bin/bash
# Usage: ./run_alphapose.sh input.mp4 output_dir

IMAGE_NAME=alphapose
DOCKERFILE=docker/alphapose/Dockerfile

# Check if the image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "Docker image '$IMAGE_NAME' not found. Building it now..."
  docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

INPUT_VIDEO=${1:-input.mp4}
OUTPUT_DIR=${2:-outputs}

docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace/alphapose \
  "$IMAGE_NAME" python scripts/demo_inference.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --video "$INPUT_VIDEO" \
    --outdir "$OUTPUT_DIR"