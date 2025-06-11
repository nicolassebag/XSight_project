  FROM python:3.12
  COPY XSight /XSight
  COPY data /data
  COPY setup.py /setup.py
  COPY Requirements_prod.txt /Requirements.txt
  RUN pip install --upgrade pip
  RUN pip install -e .
  CMD uvicorn XSight.api.fast:app --host 0.0.0.0 --port $PORT
