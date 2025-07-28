FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# copy code
COPY . /app

# install deps
RUN pip install --no-cache-dir -r requirements.txt

# default cmd: process Round1B task
# (the judging harness will mount /app/input & /app/output)
CMD ["python", "main_b.py", "--input", "/app/input", "--output", "/app/output", "--task", "task.json"]
