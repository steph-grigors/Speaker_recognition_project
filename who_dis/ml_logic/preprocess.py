import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from who_dis.ml_logic.registry import load_audio_file, save_ASR_input
from who_dis.params import *



def OHE_target(clean_df):
    '''
    This function takes in the DataFrame with information about our data.
    Return the One Hot Encoded target from the 'speaker' column of that DataFrame.
    '''
    # Instantiate the OneHotEncoder
    ohe = OneHotEncoder(sparse = False)
    # Fit encoder
    ohe.fit(clean_df[['speaker']])
    # Transform the current "Street" column
    clean_df[ohe.get_feature_names_out()] = ohe.transform(clean_df[['speaker']])
    y_encoded = np.array(clean_df[ohe.get_feature_names_out()])

    return y_encoded


def get_MFCC_features(audio, sample_rate=16000):
    '''
    mfccs_scaled_features is an array of shape (40, ) MFC coefficients that represent our features
    '''
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features


def get_MEL_spectrogram(audio, sample_rate=16000):
    '''
    This function computes the MEL spectrogram from an audio file.
    The output is a MEL spectrogram image.
    '''
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=256, n_mels=128, center = True, pad_mode = 'symmetric')
    mel_spect = np.expand_dims(mel_spect, axis = 2)
    mel_spect = librosa.util.pad_center(mel_spect, size = 606, axis = 1)

    return mel_spect


##########################################################################################################

def MFCC_features_extractor(clean_df, dataset = 'train'):

    assert dataset == 'train' or dataset == 'test'

    dir_path = os.path.dirname(os.getcwd())

    if dataset == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting MFCC feature from row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_train.append(get_MFCC_features(audio, sample_rate))

        X_train = np.array(X_train)

        return X_train

    if dataset == 'test':
        X_test = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting MFCC feature from row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_test.append(get_MFCC_features(audio, sample_rate))

        X_test = np.array(X_test)

        return X_test

##########################################################################################################

def MEL_spect_features_extractor(clean_df, dataset = 'train'):
    '''
    This function takes in the DataFrame with the information about the original data.
    It loops over it and applies the get_MEL_spectrogram() function to the audio array.
    '''
    assert dataset == 'train' or dataset == 'test'

    dir_path = os.path.dirname(os.getcwd())

    if dataset == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting MEL spect from row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            mel_spect = np.array(get_MEL_spectrogram(audio, sample_rate))
            X_train.append(mel_spect)

        X_train = np.array(X_train)

        return X_train


    if dataset == 'test':
        X_test = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting MEL spect from row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            mel_spect = np.array(get_MEL_spectrogram(audio, sample_rate))
            X_test.append(mel_spect)

        X_test = np.array(X_test)

        return X_test


def get_ASR_input(clean_df, dataset: str):
    '''
    This function takes in the DataFrame with all the info about our data with a string ('train' or 'test').
    Returns the ASR_input as DataFrame (saves it to BQ as well).
    '''
    assert dataset == 'train' or dataset == 'test'
    dir_path = os.path.dirname(os.getcwd())
    if dataset == 'train':
        tmp = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting audio files # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sr = load_audio_file(audiofile_path)
            tmp.append(audio)

        ASR_input = pd.DataFrame(clean_df['speech'])
        ASR_input['audio'] = tmp
        save_ASR_input(ASR_input,'train')

        return ASR_input

    if dataset == 'test':
        tmp = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Extracting audio files # {index} / {len(clean_df)}', end='\r')
            audiofile_path = test_set + row['file_path']
            audio, sr = load_audio_file(audiofile_path)
            tmp.append(audio)

        ASR_input = pd.DataFrame(clean_df['speech'])
        ASR_input['audio'] = tmp
        save_ASR_input(ASR_input,'test')

        return ASR_input
