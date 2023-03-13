import os


##################  VARIABLES  ##################

GCP_CREDS = os.environ.get("GCP_CREDS")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MODEL_TARGET = os.environ.get('MODEL_TARGET')

##################  PREPROCESSING  ##################
sample_rate = 16000
n_fft = 512
hop_length = 128

##################  TRAIN_MODEL  ##################
learning_rate = 0.001
batch_size=64
patience=5
validation_split=0.2
