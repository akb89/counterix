FROM python:3

ADD . /counterix

WORKDIR /counterix

RUN python -m pip install numpy

RUN python -m pip install Cython

RUN python setup.py install
