#
# An example of a Dockerfile for lusee-night/luuseepy,
# kept of illustration purposes only.
#
# Please refer to other Dockerfiles in this folder
# for working examples.
# 
# This file is to be referenced when building the luseepy

FROM python:3.10.1-bullseye

WORKDIR /app

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY lusee lusee
COPY tests tests

CMD ["python"]
