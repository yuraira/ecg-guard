FROM python:3.12-slim@sha256:9e869b0816f5537709825b49e62dc86d1c2691eff19b05c1d4dc3a07992cc052

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="ECG Guard" \
      org.opencontainers.image.description="Research-only ECG AI demonstration" \
      org.opencontainers.image.source="https://github.com/yuraira/ecg-guard" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim" \
      org.opencontainers.image.base.digest="sha256:9e869b0816f5537709825b49e62dc86d1c2691eff19b05c1d4dc3a07992cc052" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    ECG_GUARD_DEVICE=cpu \
    ECG_GUARD_CHECKPOINT=/models/best_model.pt \
    ECG_GUARD_VERIFY_CHECKPOINT_AT_STARTUP=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1=14.2.0-19 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements-container.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY scripts/generate_sbom.py ./scripts/generate_sbom.py
COPY .streamlit ./.streamlit

RUN python -m pip install --requirement requirements-container.lock \
    && python -m pip install \
      --no-deps \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.12.1 \
    && python -m pip install --no-deps --no-build-isolation . \
    && python -m pip check \
    && python ./scripts/generate_sbom.py \
      --output /app/sbom/python-runtime.cdx.json \
    && addgroup --system ecgguard \
    && adduser \
      --system \
      --ingroup ecgguard \
      --home /home/ecgguard \
      ecgguard \
    && chown -R ecgguard:ecgguard /app /home/ecgguard

USER ecgguard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

ENTRYPOINT ["ecg-guard-container-entrypoint"]

CMD ["ecg-guard-web", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
