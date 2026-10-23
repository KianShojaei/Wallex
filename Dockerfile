FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Tehran PORT=8000
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
# persistent disk (Liara) is mounted at /app/data: state, decision logs and AI cache live there
RUN mkdir -p /app/data/state /app/data/logs/decisions /app/data/logs/ai_cache /app/data/candles \
 && ln -s /app/data/state /app/state && ln -s /app/data/logs /app/logs
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=120s CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)" || exit 1
ENTRYPOINT ["wallex-agent"]
CMD ["serve"]
