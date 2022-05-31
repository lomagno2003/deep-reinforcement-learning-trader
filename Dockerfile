FROM python:3.9

WORKDIR /drltrader-bot
RUN mkdir -p logs

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ARG INCUBATOR_VER=unknown
RUN ls -l

COPY . .

EXPOSE 8080

CMD [ "python3", "run_brain.py"]