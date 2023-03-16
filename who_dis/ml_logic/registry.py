import pandas as pd
import glob
import os
import time
import pickle
from colorama import Fore, Style
import librosa
from tensorflow import keras
from who_dis.params import *
import google.auth


def load_audio_file(audiofile_path):
    '''
    audio represents the values of each of the n_samples taken at a 16Khz frequency rate
    sample_rate is set to None as to use the native sampling rate
    mono = True sets the n_channels to 1
    duration can be changed in params but is set to None by default - set the value in seconds if we only need to load  x seconds of the audiofile
    '''
    audio, sample_rate = librosa.load(audiofile_path, sr= None, mono = True, offset = 0.0, duration = None, res_type='soxr_hq')

    return audio, sample_rate

def load_cleaned_df(csv_path):
    '''
    Fetch the dataframe containing the main informations about the data.
    Takes in the filepath where the .csv file containing the dataframe can be found.
    Returns the dataframe.
    '''
    df_cleaned = pd.read_csv(csv_path)

    return df_cleaned

def save_preprocessed(X,y,data: str ) -> None:
    '''
    Saving preprocessed data in a local file or/and in a BQ dataset.
    Takes in the MFCC features, the MEL spectrograms, and the target labels.
    It also takes in a string which is to be set as 'train' or 'test'
    '''
    # save preprocessed data locally
    data_path = os.path.join(LOCAL_REGISTRY_PATH, 'prepro_data')

    if DATA_TARGET == 'local':
        X_filename = os.path.join(data_path,f'{data}/X_{data}.pickle')
        y_filename = os.path.join(data_path, f'{data}/y_{data}.pickle')
        with open(X_filename, 'wb') as handle:
            pickle.dump(X,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(y_filename, 'wb') as handle:
            pickle.dump(y,handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("✅ Results saved locally")

    if DATA_TARGET == 'gcs':
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(PREPRO_BUCKET)

        X_filename = os.path.join(data_path,f'{data}/X_{data}.pickle')
        y_filename = os.path.join(data_path, f'{data}/y_{data}.pickle')

        Xblob = bucket.blob(f'{data}/X_{data}')
        Xblob.upload_from_filename(X_filename)
        yblob = bucket.blob(f'{data}/y_{data}')
        yblob.upload_from_filename(y_filename)

        print(f"✅ Data saved to the dedicated bucket")

    return None

def load_preprocessed(data: str):
    '''
    Loads the preprocessed data from the local path or from the BQ datasets.
    Takes in which data the user wants to load 'train' or 'test'.
    Return MFCC features, MEL spectrograms and target in that order.
    '''
    data_path = os.path.join(LOCAL_REGISTRY_PATH, 'prepro_data')
    X_filename = os.path.join(data_path,f'{data}/X_{data}.pickle')
    y_filename = os.path.join(data_path, f'{data}/y_{data}.pickle')
    X = pickle.load(open(X_filename, 'rb'))
    y = pickle.load(open(y_filename, 'rb'))

    return X, y

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # save model in a dedicated gcs bucket
        from google.cloud import storage

        model_filename = model_path.split("/")[-1] # get timestamp
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to gcs")
        return None

    return None

def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        # Get latest model version name by timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")
        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        lastest_model = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ model loaded from local disk")

        return lastest_model


    elif MODEL_TARGET == "gcs":
        from google.cloud import storage

        client = storage.Client()
        blobs = list(client.get_bucket(MODEL_BUCKET).list_blobs(prefix="model"))
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        lastest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return lastest_model

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on mlflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

        # save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_ASR_input(ASR_input,dataset: str):
    from google.cloud import bigquery
    TABLE = f"ASR_df_{dataset}"
    table = f'{GCP_PROJECT}.{BQ_DATASET}.{TABLE}'
    credentials, project = google.auth.default()
    client = bigquery.Client(project,credentials)
    write_mode = "WRITE_TRUNCATE"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)
    job = client.load_table_from_dataframe(ASR_input, table, job_config=job_config)
    result = job.result()
    return None

def load_ASR_input(dataset: str):
    from google.cloud import bigquery
    TABLE = f"ASR_df_{dataset}"
    query = f"""
            SELECT *
            FROM {GCP_PROJECT}.{BQ_DATASET}.{TABLE}
            """
    credentials, project = google.auth.default()
    client = bigquery.Client(str(project),str(credentials))
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    return df
