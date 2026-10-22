FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Tehran
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
VOLUME ["/app/state", "/app/logs", "/app/data"]
ENTRYPOINT ["wallex-agent"]
CMD ["run", "--mode", "shadow"]
