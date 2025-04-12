# GaussianFormer Docker Setup

This Docker setup provides a reproducible environment for running GaussianFormer with all dependencies pre-installed.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with compatible drivers
- NVIDIA Container Toolkit (nvidia-docker)

## Quick Start

1. Build the Docker image:
   ```bash
   docker compose build
   ```

2. Run the container (interactive mode):
   ```bash
   docker run --gpus all -it --ipc=host --shm-size=16G -v "$(pwd)":/app --rm gaussianformer:latest
   ```

3. Exit the container:
   ```bash
   exit
   ```

## Docker Configuration

The setup includes:

- CUDA 11.8 with cuDNN 8
- PyTorch 2.0.0 with CUDA 11.8 support
- All MMLab packages (mmcv, mmdet, mmsegmentation, mmdet3d)
- Custom CUDA operations compiled from source
- Optional visualization packages

## Customization

- The Docker image uses a multi-stage build for better caching
- Local files are mounted to `/app` in the container
- X11 forwarding is configured for GUI applications if needed

## Troubleshooting

If you encounter issues with CUDA or GPU access:

1. Make sure NVIDIA drivers are properly installed on the host
2. Verify NVIDIA Container Toolkit is properly installed:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
   ```
3. Check if the Docker daemon configuration includes the NVIDIA runtime
