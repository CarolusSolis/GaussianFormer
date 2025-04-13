FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS builder

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
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CPATH=$CUDA_HOME/include:$CPATH
ENV LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0"

# PyTorch is already installed in the base image
# Install torchvision and torchaudio compatible with the installed torch version
RUN pip install --no-cache-dir torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMLab packages (cached as a separate layer)
RUN pip install --no-cache-dir openmim && \
    mim install mmcv==2.1.0 && \
    mim install mmdet==3.3.0 && \
    mim install mmsegmentation==1.2.2 && \
    mim install mmdet3d==1.4.0

# Install missing dependencies
RUN pip install --no-cache-dir ftfy regex jaxtyping einops wandb

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
RUN cd /app/model/head/localagg_prob && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    pip install -e . && \
    cd /app/model/head/localagg && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    pip install -e . && \
    cd /app/model/head/localagg_prob_fast && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    pip install -e . && \
    cd /app/model/encoder/gaussian_encoder/ops && \
    CPATH=$CUDA_HOME/include:$CPATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    pip install -e . && \
    cd /tmp && \
    git clone https://github.com/xieyuser/pointops.git && \
    cd pointops && \
    echo "from pointops.functions.pointops import furthestsampling as farthest_point_sampling" > __init__.py && \
    python setup.py install

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
