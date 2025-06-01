FROM python:3.10.15-slim-bullseye

WORKDIR app

COPY src src
COPY src/requirements.txt .
RUN rm -rf data
RUN rm -rf src/data

RUN ls

RUN python3 -m pip install -r requirements.txt

EXPOSE 8888