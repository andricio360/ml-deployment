
FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /usr/src/iris
WORKDIR /usr/src/iris

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "iris.app:app"]