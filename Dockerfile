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

# Cache MegaLoc architecture + weights via torch.hub at build time so the
# first runtime invocation isn't slow. (The Google variant baked weights
# from R2 during CI; we skip that here for repo simplicity — runtime fetch
# is plenty fast.)
RUN python -c "import torch; m = torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True); print('MegaLoc weights cached:', sum(p.numel() for p in m.parameters()), 'params')"

# Application code
COPY pipeline.py r2_storage.py redis_queue.py entrypoint.sh ./
COPY apple_pd/ ./apple_pd/

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
