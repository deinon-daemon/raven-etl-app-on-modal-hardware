# Use the official Python image as base image
FROM python:3.9-slim-buster
ENV OPENAI_API_KEY=
ENV GOO_MAP_TOKEN=
ENV APP_HOME /app
# Copy the app and install dependencies
WORKDIR $APP_HOME
COPY . ./
RUN pip install -r requirements.txt
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--threads", "4", "--timeout", "30", "--keep-alive", "5"]
