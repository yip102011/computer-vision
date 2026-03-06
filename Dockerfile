# ==============================
# YOLOX Docker (GPU) with small weights yolox_s.pth
# Base: PyTorch 2.4 + CUDA 12.4
# ==============================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

# Clone YOLOX repo
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git

WORKDIR /workspace/YOLOX

# This is SHA256 hash of yolox_s.pth, weight file from official github repo release 0.1.1rc0 (68.7 MB) 
ENV YOLOX_S_SHA256="f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"

# Download pretrained weights + SHA256 integrity check using the ENV var
RUN mkdir -p weights && \
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O weights/yolox_s.pth --no-verbose && \
    # Verify hash using the ENV variable (fail build on mismatch)
    echo "${YOLOX_S_SHA256} weights/yolox_s.pth" | sha256sum -c - || \
    (echo "ERROR: SHA256 mismatch for yolox_s.pth! Expected: ${YOLOX_S_SHA256}" && exit 1) && \
    ls -lh weights/yolox_s.pth && \
    echo "Weights integrity verified (SHA256 matches ${YOLOX_S_SHA256})"

# Replace origin requirement.txt file with fixed version to avoid version issue
RUN tee requirements.txt << EOF
## Core YOLOX deps – pinned to your working env
numpy==2.1.1
torch==2.4.1+cu124
opencv-python==4.13.0.92
loguru==0.7.3
tqdm==4.66.5
torchvision==0.19.1+cu124
thop==0.1.1.post2209072238
ninja==1.11.1.1
tabulate==0.9.0
tensorboard==2.20.0

## verified versions
## pycocotools corresponds to https://github.com/ppwwyyxx/cocoapi
pycocotools==2.0.11
onnx==1.20.1
onnxsim==0.4.36

Pillow==10.2.0
EOF

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install YOLOX editable (use --no-build-isolation to avoid torch import issues during build)
RUN pip install -v -e . --no-build-isolation

# quick test that weights load
RUN python -c "import torch; ckpt = torch.load('weights/yolox_s.pth', map_location='cpu'); print('Weights loaded OK, keys:', len(ckpt.keys()))"

# Suppress the specific torch.load FutureWarning globally for all python runs
ENV PYTHONWARNINGS="ignore:You are using `torch.load` with `weights_only=False`:FutureWarning"

# Default: interactive shell
CMD ["bash"]