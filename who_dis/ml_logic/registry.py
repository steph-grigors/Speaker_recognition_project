import pandas as pd
import glob
import os
import time
import pickle
from colorama import Fore, Style
import librosa
from tensorflow import keras
from who_dis.params import *

def load_audio_file(audiofile_path):
    '''
    audio represents the values of each of the n_samples taken at a 16Khz frequency rate
    sample_rate is set to None as to use the native sampling rate
    mono = True sets the n_channels to 1
    '''
    audio, sample_rate = librosa.load(audiofile_path, sr= None, mono = True, offset = 0.0, duration = 6.0, res_type='soxr_hq')

    return audio, sample_rate

def load_cleaned_df(csv_path):
    '''
    Fetch the dataframe containing the main informations about the data.
    Takes in the filepath where the .csv file containing the dataframe can be found.
    Returns the dataframe.
    '''
    df_cleaned = pd.read_csv(csv_path)
    return df_cleaned

def save_preprocessed(MFCC_feat, MEL_spec,target,data: str ) -> None:
    '''
    Saving preprocessed data in a local file or/and in a BQ dataset.
    Takes in the MFCC features, the MEL spectrograms, and the target labels.
    It also takes in a string which is to be set as 'train' or 'test'
    '''
    # save preprocessed data locally
    data_path = os.path.join(LOCAL_REGISTRY_PATH, 'prepro_data')

    preprocessed_data = [MFCC_feat, MEL_spec,target]
    paths = []
    paths.append(os.path.join(data_path,f'MFCC_features_{data}.pickle'))
    paths.append(os.path.join(data_path,f'MEL_spectrograms_{data}.pickle'))
    paths.append(os.path.join(data_path, f'labeled_target_{data}.pickle'))

    for data,path in zip(preprocessed_data,paths):
        with open(path) as file:
            pickle.dump(data,file)

    print("✅ Results saved locally")

    if DATA_TARGET == 'bq':
        from google.cloud import bigquery
        client = bigquery.Client()
        write_mode = "WRITE_TRUNCATE"
        job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

        MFCC_table = f'{GCP_PROJECT}.{BQ_DATASET}.MFCC_features_{data}'
        MEL_spectrogram_table = f'{GCP_PROJECT}.{BQ_DATASET}.MEL_spectrograms_{data}'
        target_table = f'{GCP_PROJECT}.{BQ_DATASET}.labeled_target_{data}'

        job1 = client.load_table_from_dataframe(MFCC_feat, MFCC_table, job_config=job_config)
        result = job1.result()
        job2 = client.load_table_from_dataframe(MEL_spec, MEL_spectrogram_table, job_config=job_config)
        result = job2.result()
        job3 = client.load_table_from_dataframe(target, target_table, job_config=job_config)
        result = job3.result()

        print(f"✅ Data saved to bigquery.")

    return None

def load_preprocessed(data: str):
    '''
    Loads the preprocessed data from the local path or from the BQ datasets.
    Takes in which data the user wants to load 'train' or 'test'.
    Return MFCC features, MEL spectrograms and target in that order.
    '''
    if DATA_TARGET == 'local':
        data_path = os.path.join(LOCAL_REGISTRY_PATH, 'prepro_data')
        MFCC_feat = pickle.load(os.path.join(data_path,f'MFCC_features_{data}.pickle'))
        MEL_spectrograms = pickle.load(os.path.join(data_path,f'MEL_spectrograms_{data}.pickle'))
        target = pickle.load(os.path.join(data_path, f'labeled_target_{data}.pickle'))

        return MFCC_feat, MEL_spectrograms, target

    elif DATA_TARGET == 'bq':
        from google.cloud import bigquery as bq
        MFCC_table = f'{GCP_PROJECT}.{BQ_DATASET}.MFCC_features_{data}'
        MEL_spectrogram_table = f'{GCP_PROJECT}.{BQ_DATASET}.MEL_spectrograms_{data}'
        target_table = f'{GCP_PROJECT}.{BQ_DATASET}.labeled_target_{data}'

        client = bq.Client(GCP_PROJECT)
        query_job1 = client.query(f'''
                                SELECT *
                                FROM {MFCC_table}
                                ''')
        result1 = query_job1.result()
        MFCC_feat = result1.to_dataframe()
        query_job2 = client.query(f'''
                                SELECT *
                                FROM {MEL_spectrogram_table}
                                ''')
        result2 = query_job2.result()
        MEL_spectrograms = result2.to_dataframe()
        query_job3 = client.query(f'''
                                SELECT *
                                FROM {target_table}
                                ''')
        result3 = query_job3.result()
        target = result3.to_dataframe()

        print(f"✅ Data loaded.")

        return MFCC_feat, MEL_spectrograms, target

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
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

def load_model(stage="Production") -> keras.Model:
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
