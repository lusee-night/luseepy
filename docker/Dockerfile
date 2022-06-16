##### USED PRIMARILY TO BUILD THE FOUNDATION IMAGE
#
# NB. the "reqs" build argument can supply any  requirements file for
# the build, defaulting to the common "requirements.txt"
# 
# Also note a copy of the "astropy cache"

FROM python:3.10.1-bullseye

ARG reqs=requirements.txt

WORKDIR /data

WORKDIR /app

COPY docker/${reqs} requirements.txt
RUN pip install -r requirements.txt

RUN rm requirements.txt

COPY lusee lusee
COPY simulation simulation
COPY tests tests

ADD .astropy /root/.astropy

CMD ["python"]
