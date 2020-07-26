FROM python:3

ADD . /counterix

WORKDIR /counterix

RUN python3 setup.py install
