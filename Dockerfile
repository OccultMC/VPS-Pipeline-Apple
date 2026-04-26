FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System dependencies (libheif for HEIC decode in pillow_heif)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl dos2unix build-essential python3-dev \
    libheif-dev && \
    rm -rf /var/lib/apt/lists/*

# Install vastai CLI for self-destruct
RUN pip install --no-cache-dir vastai

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch pyequilib's empty __init__.py so streetlevel.lookaround imports cleanly
# (the 0.5.8 wheel ships the submodule files but no top-level re-export)
RUN python -c "import equilib, os; p = os.path.join(os.path.dirname(equilib.__file__), '__init__.py'); open(p, 'w').write('from equilib.equi2equi.base import Equi2Equi  # noqa: F401\\n')" && \
    python -c "from equilib import Equi2Equi; from streetlevel.lookaround import to_equirectangular; print('streetlevel + equilib OK')"

# Cache MegaLoc architecture (hubconf.py + model code) — no weights downloaded
RUN python -c "import torch; torch.hub._get_cache_or_reload('gmberton/MegaLoc', force_reload=False, trust_repo=True, calling_fn='load')"

# MegaLoc model weights — downloaded from R2 by CI workflow, COPYd into image
COPY models/megaloc/model.safetensors /app/models/megaloc/model.safetensors

# Application code
COPY pipeline.py r2_storage.py redis_queue.py entrypoint.sh ./
COPY apple_pd/ ./apple_pd/

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
