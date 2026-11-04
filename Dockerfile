# Earshot agent server + web client. Build from the repo root: docker build -t earshot .
# Set META_API_KEY (and optionally FIREWORKS_API_KEY) as host secrets; server/.env is never copied in.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/
WORKDIR /app
COPY server/pyproject.toml server/uv.lock server/
RUN uv --directory server sync --frozen --no-dev --no-install-project
COPY server server
COPY web web
RUN uv --directory server sync --frozen --no-dev
ENV EARSHOT_DATA_DIR=/tmp/earshot PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uv --directory server run --no-sync uvicorn earshot.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
