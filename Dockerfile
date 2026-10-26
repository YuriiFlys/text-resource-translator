FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY sql ./sql
COPY sources.yaml ./
COPY samples ./samples

RUN pip install --upgrade pip && pip install ".[sources,embeddings-local,llm-ollama]"

ENTRYPOINT []
CMD ["trt", "--help"]
