from fastapi import FastAPI
import requests

###################### / ######################

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Speaker Recogn API test': True}

###################### /predict ######################

@app.get('/predict')
def predict(day_of_week, time):
    # Compute `wait_prediction` from `day_of_week` and `time`
    wait_prediction = int(day_of_week) * int(time)

    return {'wait': wait_prediction}
