FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    LISBOA_RUNTIME_DATA_DIR=/tmp/lisboa_runtime \
    VECTOR_DB_DIR=/tmp/lisboa_runtime/vector_db \
    HF_HOME=/tmp/lisboa_runtime/huggingface \
    XDG_CACHE_HOME=/tmp/lisboa_runtime/cache \
    TZ=Europe/Lisbon \
    OTEL_SDK_DISABLED=true \
    ANONYMIZED_TELEMETRY=false \
    CHROMA_TELEMETRY=false \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "scripts/hf_space_entrypoint.py"]
