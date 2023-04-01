# The same Python version of your virtual env
# FROM image
# FROM python:3.10.6-buster
#  Let's lighted the image with tensorflow

FROM tensorflow/tensorflow:2.10.0
# All the directories from the /taxifare project needed to run the API
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

#CREDENTIAL JSON
COPY credentialsdocker.json credentials.json

# COPY api /api
# The list of dependencies (don’t forget to install them!)
COPY who_dis /who_dis
COPY setup.py setup.py
RUN pip install --upgrade pip
RUN pip install .

# TODO: add this to requirements.txt
RUN pip install soundfile
# RUN pip install librosa==0.10.0
RUN apt-get update
RUN apt-get install -y libsndfile1

RUN mkdir -p models

# #Let's connect to our running continer
# taxifare.api.fast:app est le chemain des fichiers pour aller chercher app
CMD  uvicorn who_dis.api.fast:app --host 0.0.0.0 --port $PORT
