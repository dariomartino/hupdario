FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/hup
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY config.yaml .
COPY app ./app

CMD ["python", "-m", "app.main"]
