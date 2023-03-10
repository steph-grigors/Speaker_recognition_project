import wave
import os
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def OHE_target(clean_df):

    # Instantiate the OneHotEncoder
    ohe = OneHotEncoder(sparse_output = False)
    # Fit encoder
    ohe.fit(clean_df[['speaker']])
    # Find the encoded classes
    print(f"The categories detected by the OneHotEncoder are {ohe.categories_}")
    # Transform the current "Street" column
    clean_df[ohe.get_feature_names_out()] = ohe.transform(clean_df[['speaker']])
    y_encoded = np.array(clean_df[ohe.get_feature_names_out()])

    return y_encoded

def load_audio_file(audiofile_path):
    '''audio represents the values of each of the n_samples taken at a 16Khz frequency rate
    sample_rate is set to None as to use the native sampling rate
    mono = True sets the n_channels to 1'''
    audio, sample_rate = librosa.load(audiofile_path, sr= None, mono = True, offset = 0.0, duration = 6.0, res_type='soxr_hq')

    return audio, sample_rate

def get_MFCC_features(audio, sample_rate = 16000):
    '''
    mfccs_scaled_features is an array of shape (40, ) MFC coefficients that represent our features
    '''
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features

def show_MEL_spectrogram(audio, sample_rate = 16000):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=128, center = True, pad_mode = 'symmetric')
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=16000, x_axis='time');
    plt.title('Mel Spectrogram');
    plt.colorbar(format='%+2.0f dB');


def get_MEL_spectrogram(audio, sample_rate = 16000):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=128, center = True, pad_mode = 'symmetric')
    # mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = np.expand_dims(mel_spect, axis = 2)
    mel_spect = librosa.util.pad_center(mel_spect, size = 751, axis = 1)

    return mel_spect


##########################################################################################################

def MFCC_features_extractor(clean_df, dataset = 'train'):

    assert dataset == 'train' or dataset == 'test'

    dir_path = os.path.dirname(os.getcwd())

    if dataset == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_train.append(get_MFCC_features(audio, sample_rate))

        X_train = np.array(X_train)

        return X_train

    if dataset == 'test':
        X_test = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_test.append(get_MFCC_features(audio, sample_rate))

        X_test = np.array(X_test)

        return X_test

##########################################################################################################

def MEL_spect_features_extractor(clean_df, dataset = 'train'):

    assert dataset == 'train' or dataset == 'test'

    dir_path = os.path.dirname(os.getcwd())

    if dataset == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
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
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            mel_spect = np.array(get_MEL_spectrogram(audio, sample_rate))
            X_test.append(mel_spect)

        X_test = np.array(X_test)

        return X_test




###########################################################################################################

# def clipping_unbalanced_classes(df_cleaned):
#     '''Checks for the number of observations per target class and clips
#     the audio files in 2 clips if values_count of that class < max(values_count))'''

#     for num_recordings in df_cleaned.speaker.value_counts(normalize = True):
#         if num_recordings > df_cleaned.speaker.value_counts().max() // 2:
#             pass #clip the audio file in 2 windows'''
###########################################################################################################

# def trim_recordings():
#     pass
###########################################################################################################
