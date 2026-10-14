# Prospecta service: HTTP entry point over a Postgres+pgvector database.
# Build:  docker build -t prospecta-service .
# Run:    docker run -p 8080:8080 -e DATABASE_URL=... -e PROSPECTA_SCHEMA=prospecta \
#           -e PROSPECTA_SOURCE_SCHEMA=... -e PROSPECTA_SERVICE_TOKEN=... prospecta-service
FROM python:3.12-slim
WORKDIR /app
RUN useradd --system --uid 10001 prospecta
COPY pyproject.toml README.md LICENSE ./
COPY prospecta ./prospecta
RUN pip install --no-cache-dir ".[defaults]"
USER prospecta
ENV PROSPECTA_SERVICE_HOST=0.0.0.0 PROSPECTA_SERVICE_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ['PROSPECTA_SERVICE_PORT']+'/healthz')"
CMD ["python", "-m", "prospecta.service"]
