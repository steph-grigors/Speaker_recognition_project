import os


##################  VARIABLES  ##################

GCP_CREDS = os.environ.get("GCP_CREDS")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MODEL_TARGET = os.environ.get('MODEL_TARGET')
LOCAL_REGISTRY_PATH = os.path.abspath(os.path.dirname(os.getcwd()))
