#!/bin/bash
set -e

date -u
echo "=== Hypervision VPS Pipeline (Apple Look Around) ==="
echo "Instance: ${INSTANCE_ID}"
echo "City: ${CITY_NAME}"
echo "Region: ${REGION}"
echo "Redis: ${REDIS_URL:0:40}..."

# ── GPU diagnostics ──
echo "── GPU Diagnostics ──"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_CUDA=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo 'N/A')
    echo "GPU: ${GPU_NAME}"
    echo "VRAM: ${GPU_MEM}"
    echo "Driver: ${GPU_DRIVER}"
    echo "Compute Cap: ${GPU_CUDA}"
else
    echo "[ERROR] nvidia-smi not found! GPU may not be available."
fi

echo "── System Info ──"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "RAM: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'N/A') total"
echo "CPU: $(nproc) cores"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA (PyTorch): $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "Working Directory: $(pwd)"
echo "Files in /app:"
ls -la /app
echo "================================"

# Instance ID detection is handled by pipeline.py via R2 lookup.
if [ -z "${INSTANCE_ID}" ]; then
    echo "[INFO] INSTANCE_ID not set — pipeline.py will detect via R2"
fi

# Check MegaLoc model sources
MODEL_FILE="/app/models/megaloc/model.safetensors"
HUB_CACHE="/root/.cache/torch/hub/gmberton_MegaLoc_main"
echo "── MegaLoc Model ──"
echo "Primary: torch.hub get_trained_model (HuggingFace, same as inference server)"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "Fallback: baked model at $MODEL_FILE ($MODEL_SIZE)"
else
    echo "Fallback: no baked model (hub download required)"
fi
if [ -d "$HUB_CACHE" ]; then
    echo "[OK] MegaLoc architecture cached at $HUB_CACHE"
else
    echo "[INFO] MegaLoc architecture not cached — will download from GitHub"
fi

# Fix for PyTorch 2.x compile (inductor) missing libcuda.so
# Find where libcuda.so.1 is (mounted by nvidia-runtime) and symlink libcuda.so
echo "[INFO] Checking for libcuda.so..."
LIBCUDA_PATH=$(ldconfig -p | grep libcuda.so.1 | head -n 1 | awk '{print $4}')
if [ -n "$LIBCUDA_PATH" ]; then
    DIR=$(dirname "$LIBCUDA_PATH")
    if [ ! -f "$DIR/libcuda.so" ]; then
        echo "[INFO] Creating libcuda.so symlink at $DIR/libcuda.so -> $LIBCUDA_PATH"
        ln -s "$LIBCUDA_PATH" "$DIR/libcuda.so"
    else
        echo "[INFO] libcuda.so already exists at $DIR/libcuda.so"
    fi
else
    echo "[WARN] Could not find libcuda.so.1 in ldconfig cache. Compilation might fail."
    # Try common location fallback
    if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
         echo "[INFO] Fallback: Creating symlink in /usr/lib/x86_64-linux-gnu/"
         ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
    fi
fi

# Suppress Vast.ai SSH port-forwarding noise from polluting pipeline logs
# These spam "Warning: Permanently added" + "remote port forwarding failed" every 2s
export VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-}
(
    # Kill the noisy ssh port-forward retry loop if it exists
    # Redirect any remaining sshd/ssh stderr to /dev/null
    for pid in $(pgrep -f "ssh.*vast.ai" 2>/dev/null); do
        kill "$pid" 2>/dev/null || true
    done
) &>/dev/null || true

# Run the pipeline — filter out SSH noise from stdout/stderr
echo "[INFO] Starting pipeline.py at $(date -u)..."
python pipeline.py 2>&1 | grep -v -E "^(Warning: Permanently added|Error: remote port forwarding|kex_exchange_identification|Connection closed by|Connection from|banner exchange|ssh: Could not resolve)" || true
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Pipeline exited with code $EXIT_CODE at $(date -u)"
    echo "[INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
fi
