FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libchromaprint-tools \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY app ./app

RUN pip install --no-cache-dir -e ".[janitor,lyrics]"

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} bardo 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} bardo 2>/dev/null || true \
    && mkdir -p /data /music /config \
    && chown -R ${UID}:${GID} /app /data /music /config

USER ${UID}:${GID}

ENV DATA_DIR=/data \
    BEETSDIR=/config \
    MUSIC_DIR=/music \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/api/health > /dev/null || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
