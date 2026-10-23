FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Tehran PORT=8000 DATA_DIR=/app/data
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir . && mkdir -p /app/data /app/logs
# Liara mounts the persistent disk at /app/data (see liara.json); DATA_DIR puts state, decision logs,
# AI cache and candle cache there so nothing is lost on restart/redeploy.
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)" || exit 1
ENTRYPOINT ["wallex-agent"]
CMD ["serve"]
