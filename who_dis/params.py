import os

##################  VARIABLES  ##################

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
PREPRO_BUCKET = os.environ.get('PREPRO_BUCKET')
MODEL_TARGET = 'gcs'
BQ_DATASET = os.environ.get("BQ_DATASET")
DATA_TARGET = 'bq'


##################  PATHS  ##################
LOCAL_REGISTRY_PATH = os.path.abspath(os.getcwd())
# LOCAL_REGISTRY_PATH = os.path.abspath(os.path.dirname(os.getcwd()))

