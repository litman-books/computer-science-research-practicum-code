# Dockerfile — highest-fidelity environment layer (Chapter 9, §9.3).
# Use this layer when your result depends on OS-level libraries (CUDA runtime,
# system gcc/libstdc++, ffmpeg) that pip freeze / conda cannot capture.
# Don't use it for a 10-second scikit-learn study — that is ceremony, not rigor.
#
#   docker build -t my-research-project .
#   docker run --rm -e CUBLAS_WORKSPACE_CONFIG=:4096:8 my-research-project

# Pick the newest Python that all your libraries ship wheels for, and pin it
# (better still, pin the image digest: python:3.14-slim@sha256:...).
FROM python:3.14-slim

# Pin OS-level packages too if your result depends on them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code and config (data is mounted / downloaded, never baked in).
COPY src/ ./src/

# Both must be set before the Python process starts; setting them from inside
# Python is too late. PYTHONHASHSEED fixes string hashing (set order);
# CUBLAS_WORKSPACE_CONFIG lets torch.use_deterministic_algorithms(True) run
# on CUDA >= 10.2 instead of raising a RuntimeError.
ENV PYTHONHASHSEED=1337
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRYPOINT ["python", "src/run_experiment.py"]
CMD ["--config", "src/config.yaml", "--seed", "1337"]
