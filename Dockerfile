# The same Python version of your virtual env
FROM image
FROM python:3.10.6-buster
#  Let's lighted the image with tensorflow

FROM tensorflow/tensorflow:2.10.0
# All the directories from the /taxifare project needed to run the API
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# COPY api /api
# The list of dependencies (don’t forget to install them!)
COPY who_dis /who_dis
COPY setup.py setup.py
RUN pip install --upgrade pip

# We already have a make command for that!
COPY Makefile Makefile
# RUN make reset_local_files

# #Let's connect to our running continer
# taxifare.api.fast:app est le chemain des fichiers pour aller chercher app
CMD  uvicorn who_dis.api.fast:app --host 0.0.0.0 --port $PORT
