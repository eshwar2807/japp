# Runtime for the hosted dashboard.
#
# Chromium is included because tailoring fetches the job description from the
# posting URL, which usually needs JavaScript. It is used headlessly and only
# for that. Application form-filling is NOT run here — it needs a browser window
# a human can watch, so those jobs are left for the local agent (JP_WORKER_KINDS).

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# WeasyPrint needs Pango/GDK-PixBuf; fonts-dejavu keeps generated PDFs legible.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libfribidi0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-dejavu-core \
        fonts-liberation \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium plus its own system libraries.
RUN playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# Run as an unprivileged user; /data is the mounted volume.
RUN useradd --create-home --uid 10001 app && \
    mkdir -p /data && chown -R app:app /app /data /opt/playwright
USER app

ENV JP_DATA_DIR=/data \
    JP_OUTPUT_DIR=/data/output \
    JP_DB_URL=sqlite:////data/pipeline.db \
    JP_KEY_PATH=/data/vault.key \
    JP_SECRET_KEY_PATH=/data/secret.key \
    JP_BROWSER_PROFILE=/data/browser_profile \
    JP_WEB_HOST=0.0.0.0 \
    JP_WEB_PORT=8080 \
    JP_WORKER_KINDS=tailor,discover

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8080"]
