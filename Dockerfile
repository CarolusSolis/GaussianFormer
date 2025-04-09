FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Update pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CPATH=$CUDA_HOME/include:$CPATH
ENV LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Install PyTorch with CUDA 11.8 (cached as a separate layer)
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install MMLab packages (cached as a separate layer)
RUN pip install --no-cache-dir openmim && \
    mim install mmcv==2.0.1 && \
    mim install mmdet==3.0.0 && \
    mim install mmsegmentation==1.0.0 && \
    mim install mmdet3d==1.1.1

# Install other packages (cached as a separate layer)
RUN pip install --no-cache-dir spconv-cu117 timm

# Install visualization packages (optional, cached as a separate layer)
RUN pip install --no-cache-dir pyvirtualdisplay mayavi matplotlib==3.7.2 PyQt5

# Final stage - copy the code and install custom ops
FROM builder AS final

# Set working directory
WORKDIR /app

# Copy the entire GaussianFormer repository
COPY . /app/

# Install custom CUDA ops with proper environment setup
RUN cd /app/model/encoder/gaussian_encoder/ops && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0" \
    pip install -e . && \
    cd /app/model/head/localagg && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0" \
    pip install -e . && \
    cd /app/model/head/localagg_prob && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0" \
    pip install -e . && \
    cd /app/model/head/localagg_prob_fast && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0" \
    pip install -e .

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
