#
# NB. the "reqs" build argument can supply any
# requirements file for the build, defaulting to the
# common "requirements.txt"
# 
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
