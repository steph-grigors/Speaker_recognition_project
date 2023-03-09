import wave
import os
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from who_dis.ml_logic.registry import load_cleaned_df

def cleaned_df(df_raw):

    """ This function takes the train.csv file (in the form of a dataframe) as argument.
    The path must may be adapted to absolute path
    The for loop iterate over all rows of the DataFrame train.csv :
    - The way is created by joining the initiale path(TBM) and the path of each files which is the column
    'file_path' of the train.csv DataFrame.
    - Calculation of each paramaters to plot the Signal Amplitude and the Frequency Spectrum
    - Plot
    """

    path = os.path.dirname(os.getcwd())
    output_df_cleaned_path = os.path.join(path,'raw_data','output_df_cleaned_preprocessing.csv')
    csv_path = Path(output_df_cleaned_path)

    #Initiate empty lists
    list_n_samples =[]
    list_t_audio =[]
    list_signal_array=[]
    list_extracted_features = []

    #Sample_frequency
    sample_rate = 16000

    i = 0
    tot = len(df_raw)

    if csv_path.is_file():
        df_cleaned = load_cleaned_df(output_df_cleaned_path)

    else:
        for row in range(len(df_raw)):
            i += 1
            print(f"Treating {i} / {tot}", end='\r')
            filename = os.path.join(path,'raw_data',df_raw['file_path'][row])
            with wave.open(filename, 'rb') as wav_obj:
                # Check the number of channels (e.g file recorder in stereo has 2 indepedent audio channels
                # (has 2 channels). This crereates the impression of the sound coming from two different directions)
                n_samples = wav_obj.getnframes()
                # how long our audio file is in seconds
                t_audio = n_samples/sample_rate
                # the amplitude of the wave at that point in time
                signal_wave = wav_obj.readframes(n_samples)
                # Turn signal wave into numpy array to get signal values from this
                signal_array = np.frombuffer(signal_wave, dtype=np.int16)
                # Audio represents the values in float of each of the n_samples
                audio, sample_rate = load_audio_file(filename)
                # Extracting features from the audiofiles
                features=get_MFCC_features(audio, sample_rate)

            # append respective lists of different values determinated above
            list_n_samples.append(n_samples)
            list_t_audio.append(t_audio)
            list_signal_array.append(signal_array)
            list_extracted_features.append(features)

        # Convert all lists into a column of DataFrame
        df_raw['n_samples'] = list_n_samples
        df_raw['t_audio'] = list_t_audio
        df_raw['signal_array'] = list_signal_array
        df_raw['MFCC_features'] = list_extracted_features

        # Create an 'Amplitude' column
        mins = df_raw["signal_array"].apply(np.min)
        maxs = df_raw["signal_array"].apply(np.max)
        df_raw['amplitude'] = np.abs(maxs - mins)


        # Dropping unnecessary columns
        df_cleaned = df_raw.drop(columns=['id', 'signal_array'])

        # Checking whether the file is already saved locally, otherwise save it.
        if not csv_path.is_file():
            df_cleaned.to_csv(output_df_cleaned_path)

    return df_cleaned

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

    return mel_spect


##########################################################################################################

def MFCC_features_extractor(clean_df, set = 'train'):

    dir_path = os.path.dirname(os.getcwd())

    if set == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_train.append(get_MFCC_features(audio, sample_rate))

        X_train = np.array(X_train)

    if set == 'test':
        X_test = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_test.append(get_MFCC_features(audio, sample_rate))

    return X_train

##########################################################################################################

def MEL_spect_features_extractor(clean_df, set = 'train'):

    dir_path = os.path.dirname(os.getcwd())

    if set == 'train':
        X_train = []
        train_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}', end='\r')
            audiofile_path = train_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_train.append(get_MEL_spectrogram(audio, sample_rate))

        X_train = np.array(X_train, ndmin=3)


    if set == 'test':
        X_test = []
        test_set = os.path.join(dir_path,'raw_data/')
        for index, row in clean_df.iterrows():
            print(f'Treating row # {index} / {len(clean_df)}')
            audiofile_path = test_set + row['file_path']
            audio, sample_rate = load_audio_file(audiofile_path)
            X_test.append(get_MEL_spectrogram(audio, sample_rate))

    return X_train


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
