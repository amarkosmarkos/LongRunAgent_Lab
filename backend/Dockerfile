FROM python:3.12-slim
WORKDIR /app
# git is not optional: each run's version DAG is a real repository built with
# git plumbing (app/gitrepo.py). Without it the substrate silently disables.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
