FROM python:3.10.15-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping  curl wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR app

COPY src src
COPY src/requirements.txt .
RUN rm -rf data
RUN rm -rf src/data

RUN ls

RUN python3 -m pip install -r requirements.txt

EXPOSE 8888
