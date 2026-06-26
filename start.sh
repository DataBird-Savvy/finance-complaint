#!/bin/sh

airflow db init

airflow users create \
  --username jiya \
  --password password \
  --firstname jiya \
  --lastname joseph \
  --role Admin \
  --email josephjiyamary@gmail.com || true

nohup airflow scheduler &

airflow webserver