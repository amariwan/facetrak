# FaceTrak headless production image (multi-arch: amd64 + arm64).
# GUI-free: tkinter is never imported by the headless entrypoint.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACETRAK_CONFIG=/data/config.json \
    FACETRAK_DB=/data/facetrak.db \
    FACETRAK_DATA=/data

# Runtime libs for opencv-contrib-python + mediapipe wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY facetrak ./facetrak
RUN pip install --upgrade pip && pip install . \
    && python -c "import facetrak.servers.headless; print('import ok')"

RUN useradd -m -u 10001 facetrak && mkdir -p /data && chown facetrak:facetrak /data
USER facetrak
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4)" || exit 1

ENTRYPOINT ["python", "-m", "facetrak.servers.headless"]
