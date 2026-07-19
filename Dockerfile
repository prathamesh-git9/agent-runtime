FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser
COPY --from=builder /install /usr/local
COPY src ./src

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD \
  python -c "import urllib.request as u; u.urlopen('http://127.1:8000/healthz').read()"

CMD ["python", "-m", "uvicorn", "agent_runtime.__main__:app", \
  "--host", "0.0.0.0", "--port", "8000"]
