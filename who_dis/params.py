import os

##################  VARIABLES  ##################

GCP_CREDS = os.environ.get("GCP_CREDS")
GCP_PROJECT = 'speaker_recognition'
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MODEL_BUCKET = 'speaker_reco_models'
PREPRO_BUCKET = os.environ.get('PREPRO_BUCKET')
MODEL_TARGET = 'gcs'
BQ_DATASET = 'speaker_reco_prepro'
DATA_TARGET = 'bq'

##################  PATHS  ##################
LOCAL_REGISTRY_PATH = os.path.abspath(os.path.dirname(os.getcwd()))

##################  PREPROCESSING  ##################

duration = None
sample_rate = 16000
n_fft = 512
hop_length = 256
n_mels = 128

##################  TRAIN_MODEL  ##################
learning_rate = 0.001
batch_size=64
patience=5
validation_split=0.2
