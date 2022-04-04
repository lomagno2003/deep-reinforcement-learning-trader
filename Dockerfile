FROM python:3.9

WORKDIR /drltrader-bot
RUN mkdir -p logs

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ARG INCUBATOR_VER=unknown
RUN ls -l

COPY . .

EXPOSE 8080

CMD [ "python3", "run_brain.py"]