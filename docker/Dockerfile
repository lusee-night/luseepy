#
# An example of a Dockerfile for lusee-night/luseepy,
# kept of illustration purposes only.
#
# Please refer to other Dockerfiles in this folder
# for working examples.
# 

FROM python:3.10.1-bullseye

ARG reqs=requirements.txt
WORKDIR /app

COPY docker/${reqs} requirements.txt
RUN pip install -r requirements.txt

ADD .astropy /root/.astropy

#COPY lusee lusee
#COPY tests tests

CMD ["python"]
