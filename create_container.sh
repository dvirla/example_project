#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
CONTAINER_NAME="${1:-python311-hf}"

# On DGX, /home quota is too small for large CUDA images (~7 GB uncompressed).
# Point ENROOT_DATA_PATH to a scratch filesystem with sufficient space.
# Override by setting the env var before calling this script, e.g.:
#   ENROOT_DATA_PATH=/scratch/$USER ./create_container.sh
ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/scratch/${USER}}"
export ENROOT_DATA_PATH
mkdir -p "${ENROOT_DATA_PATH}"

echo "[info] ENROOT_DATA_PATH=${ENROOT_DATA_PATH}"
echo "[info] Available space: $(df -h "${ENROOT_DATA_PATH}" | awk 'NR==2{print $4}') free"

SQSH_FILE="${ENROOT_DATA_PATH}/${CONTAINER_NAME}.sqsh"

# ---------------------------------------------------------------------------
# Step 1: Import base Docker image → squashfs
# ---------------------------------------------------------------------------
echo "[1/3] Importing base image: ${BASE_IMAGE}"
enroot import --output "${SQSH_FILE}" "docker://${BASE_IMAGE}"

# ---------------------------------------------------------------------------
# Step 2: Create enroot container from squashfs
# ---------------------------------------------------------------------------
echo "[2/3] Creating container: ${CONTAINER_NAME}"

# Remove existing container with same name if present
if enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "  Container '${CONTAINER_NAME}' already exists — removing it first."
    enroot remove "${CONTAINER_NAME}"
fi

enroot create --name "${CONTAINER_NAME}" "${SQSH_FILE}"

# ---------------------------------------------------------------------------
# Step 3: Provision the container
# ---------------------------------------------------------------------------
echo "[3/3] Provisioning container (Python 3.11 + HuggingFace + uv)"

enroot start --rw "${CONTAINER_NAME}" bash -c '
set -euo pipefail

# --- System packages -------------------------------------------------------
apt-get update
apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    build-essential \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    ninja-build

# --- Python 3.11 via deadsnakes PPA ----------------------------------------
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils

# Make python3.11 the default python / python3
update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Ensure pip for 3.11
curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11

# --- uv --------------------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh

# uv installer drops the binary in ~/.local/bin; expose it system-wide
UV_BIN="$(find /root -name uv -type f 2>/dev/null | head -n1)"
if [ -n "${UV_BIN}" ]; then
    cp "${UV_BIN}" /usr/local/bin/uv
    chmod +x /usr/local/bin/uv
fi

# --- PyTorch (CUDA 12.4) ---------------------------------------------------
python3.11 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# --- HuggingFace ecosystem -------------------------------------------------
python3.11 -m pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    tokenizers \
    huggingface_hub \
    peft \
    trl \
    evaluate \
    safetensors \
    sentencepiece \
    protobuf

# --- Cleanup ---------------------------------------------------------------
apt-get clean
rm -rf /var/lib/apt/lists/*
'

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Container '${CONTAINER_NAME}' is ready."
echo ""
echo "  Start an interactive shell:  enroot start --rw ${CONTAINER_NAME} bash"
echo "  Run a script:                enroot start --rw ${CONTAINER_NAME} python3 your_script.py"
echo ""
echo "  To mount a host directory add:  --mount /host/path:/container/path"
