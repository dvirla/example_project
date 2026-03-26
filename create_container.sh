#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
CONTAINER_NAME="${1:-python311-hf}"

# SQSH_PATH: where the .sqsh image is stored.
# MUST be on a shared filesystem visible to all compute nodes (NFS/Lustre/GPFS).
# Common locations on DGX clusters: /lustre/$USER, /nfs/$USER, /home/$USER.
# Override: SQSH_PATH=/lustre/$USER ./create_container.sh
SQSH_PATH="${SQSH_PATH:-/home/${USER}}"
mkdir -p "${SQSH_PATH}"
SQSH_FILE="${SQSH_PATH}/${CONTAINER_NAME}.sqsh"

# ENROOT_DATA_PATH: where enroot extracts the container during provisioning.
# Needs ~7 GB of free space; can be node-local scratch.
# Override: ENROOT_DATA_PATH=/scratch/$USER ./create_container.sh
ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/scratch/${USER}}"
export ENROOT_DATA_PATH
mkdir -p "${ENROOT_DATA_PATH}"

echo "[info] SQSH_FILE=${SQSH_FILE}  (shared, used by srun/pyxis)"
echo "[info] ENROOT_DATA_PATH=${ENROOT_DATA_PATH}  (local, used during provisioning)"
echo "[info] Shared path free space:  $(df -h "${SQSH_PATH}"       | awk 'NR==2{print $4}')"
echo "[info] Scratch path free space: $(df -h "${ENROOT_DATA_PATH}" | awk 'NR==2{print $4}')"

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

enroot start --root --rw --env NVIDIA_VISIBLE_DEVICES=void "${CONTAINER_NAME}" bash -c '
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
echo "  Via srun / pyxis (recommended on DGX):"
echo "    srun --pty --container-image=${SQSH_FILE} \\"
echo "         --container-mounts=/home/\${USER}/example_project:/workspace \\"
echo "         bash -i"
echo ""
echo "  Via enroot directly (login node only):"
echo "    enroot start --root --rw --env NVIDIA_VISIBLE_DEVICES=all ${CONTAINER_NAME} bash"
echo ""
echo "  NOTE: The .sqsh file at ${SQSH_FILE} must be on a shared filesystem"
echo "        (NFS/Lustre) accessible from all compute nodes."
