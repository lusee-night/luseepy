# This file is to be referenced when building the luseepy
# images from a folder one level above.

FROM python:3.10.1-bullseye

WORKDIR /app

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["python",]
