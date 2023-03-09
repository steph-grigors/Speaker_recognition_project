from fastapi import FastAPI
import requests


# import os
# path = os.path.dirname(os.getcwd())
# path_abs = os.path.join(path,'raw_data/who_dis/api')

###################### / ######################

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Speaker Recogn API test': True}

###################### /predict ######################

# url = 'http://localhost:8000/predict'

# params = {
#     'day_of_week': 0, # 0 for Sunday, 1 for Monday, ...
#     'time': '14:00'
# }

# response = requests.get(url, params=params)
# response.json() #=> {wait: 64}

@app.get('/predict')
def predict(day_of_week, time):
    # Compute `wait_prediction` from `day_of_week` and `time`
    wait_prediction = int(day_of_week) * int(time)

    return {'wait': wait_prediction}
