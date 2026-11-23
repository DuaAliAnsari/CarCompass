FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
# Only the served models (scripts/export_models.py), not everything in models/
COPY deploy/models/ models/

# Fail the image build if the served artifacts are missing (common when
# models/ was empty because *.joblib was gitignored on the host).
RUN python -c "from pathlib import Path; import sys; need=['models/manifest.json','models/price_global_ab.joblib','models/price_global_ab.json','models/price_new_cars_2010plus.joblib','models/price_new_cars_2010plus.json']; missing=[p for p in need if not Path(p).is_file()]; sys.exit('Docker build missing served models: ' + ', '.join(missing)) if missing else print('Served models present')"

# Set at build time so /health and /metrics report the image tag: --build-arg APP_VERSION=v2.0
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health', timeout=4)"

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
