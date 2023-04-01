from fastapi import FastAPI, UploadFile, File
from colorama import Fore, Style
import requests
import numpy as np
from who_dis.interface.main import pred
from who_dis.ml_logic.registry import load_preprocessed, load_audio_file, load_model
from who_dis.ml_logic.preprocess import get_MEL_spectrogram
from who_dis.ml_logic.ASR import get_audio_array, get_transcript
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import io
import joblib
import tensorflow as tf

###################### / ######################


app = FastAPI()
model = load_model()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
ASRmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Speaker Recogn API test': False}

###################### /predict ######################
#ex : http://localhost:8000/predict?day_of_week=0&time=14

# @app.get('/predict')
# def pred(audiofile):
@app.post('/predict')
async def pred(wav: UploadFile=File(...)):
    ### Receiving and decoding the image
    """
    Make a prediction using the latest trained model
    """
    audiofile = io.BytesIO(await wav.read())
    # audiofile = wav
    print("\n:étoile:️ Use case: predict")
    assert model is not None
    # Preprocessing the audiofile
    audio_pred, sample_rate_pred = load_audio_file(audiofile)
    X_pred = get_MEL_spectrogram(audio_pred, sample_rate_pred)
    X_pred = X_pred.reshape((-1,128,606,1))
    speaker_names = {0: 'Andrew',
                     1: 'Maximilian',
                     2: 'Parul',
                     3: 'Mike',
                     4: 'Arya',
                     5: 'Henry',
                     6: 'Chloe',
                     7: 'Laura',
                     8: 'Samuel',
                     9: 'Krish',
                     10: 'Jim',
                     11: 'Alex',
                     12: 'Kalindi',
                     13: 'Elena',
                     14: 'Walter White',
                     15: 'Jules',
                     16: 'Pascaline',
                     17: 'Kamilla'}
    # Computing y_pred and the speaker's name
    y_pred = model.predict(X_pred)
    y_pred = int(np.argmax(y_pred))
    name_pred = speaker_names[y_pred]
    print(f"\n:coche_blanche: prediction done: {y_pred} \n")
    print(f"\n:coche_blanche: The person whom voice you heard is: {name_pred} \n")
    return {
        'id': y_pred,
        'name': name_pred
    }

@app.post('/transcript')
async def trans(wav: UploadFile=File(...)):
    '''
    Returns the text (as string) being read in the input wav file.
    '''
    audiofile = io.BytesIO(await wav.read())
    array = get_audio_array(audiofile)
    transcript = get_transcript(array)
    return {
        'text': transcript
    }
