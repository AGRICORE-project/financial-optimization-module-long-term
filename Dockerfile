FROM python:3.11

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT celery -A model.algorithm_ABM worker --autoscale 10 --loglevel=info