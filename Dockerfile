FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY README.md LICENSE ./

RUN pip install --upgrade pip setuptools wheel && pip install -e ".[dev]"

ENTRYPOINT ["gtt-train"]
CMD ["--config", "configs/default.yaml", "--demo"]
