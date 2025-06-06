  # FROM python:3.12
  FROM tensorflow/tensorflow:latest
  COPY XSight /XSight
  # COPY app /app
  COPY Requirements_prod.txt /Requirements_prod.txt
  RUN pip install --upgrade pip
  RUN pip install -r Requirements_prod.txt
  CMD uvicorn XSight.api.fast:app --host 0.0.0.0 --port 8080
