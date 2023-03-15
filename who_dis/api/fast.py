from fastapi import FastAPI
import requests
from who_dis.interface.main import pred
from tensorflow.io import read_file
from tensorflow.audio import decode_wav

###################### / ######################

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Speaker Recogn API test': False}

###################### /predict ######################
#ex : http://localhost:8000/predict?day_of_week=0&time=14

@app.get('/predict')
def predict(sound):
    # Compute `wait_prediction` from `day_of_week` and `time`
    coded_wav = read_file(sound)
    plot_sound = decode_wav(coded_wav)

    # type()

    return {'wait': plot_sound}
