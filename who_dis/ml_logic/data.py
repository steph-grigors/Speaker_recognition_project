import wave
import os
import numpy as np
import pandas as pd
from pathlib import Path
from who_dis.ml_logic.registry import load_cleaned_df
from who_dis.ml_logic.preprocess import load_audio_file, OHE_target
from sklearn.preprocessing import OneHotEncoder
from who_dis.params import *


def eda_df(df_raw):

    """ This function takes the train.csv file (in the form of a dataframe) as argument.
    The path must may be adapted to absolute path
    The for loop iterate over all rows of the DataFrame train.csv :
    - The way is created by joining the initiale path(TBM) and the path of each files which is the column
    'file_path' of the train.csv DataFrame.
    - Calculation of each paramaters to plot the Signal Amplitude and the Frequency Spectrum
    - Plot
    """

    path = os.path.dirname(os.getcwd())

    #Initiate empty lists
    list_n_channels=[]
    list_sample_freg =[]
    list_n_samples =[]
    list_t_audio =[]
    list_signal_wave=[]
    list_signal_array=[]
    list_times=[]

    i = 0
    tot = len(df_raw)

    for row in range(len(df_raw)):
        i += 1
        print(f"Treating {i} / {tot}", end='\r')
        filename = os.path.join(path,'raw_data',df_raw['file_path'][row])
        with wave.open(filename, 'rb') as wav_obj:
            # Check the number of channels (e.g file recorder in stereo has 2 indepedent audio channels
            # (has 2 channels). This crereates the impression of the sound coming from two different directions)
            n_channels  = wav_obj.getnchannels()
            # The sampling rate quantifies how many samples of the sound are taken every second
            sample_freq = wav_obj.getframerate()
            # The number of individual frames, or samples, is given by
            n_samples = wav_obj.getnframes()
            # how long our audio file is in seconds
            t_audio = n_samples/sample_freq
            # the amplitude of the wave at that point in time
            signal_wave = wav_obj.readframes(n_samples)
            # Turn signal wave into numpy array to get signal values from this
            signal_array = np.frombuffer(signal_wave, dtype=np.int16)
            # Need to calculate the time at which each sample is taken before plotting signal values
            times = np.linspace(0, n_samples/sample_freq, num=n_samples)

        # append respective lists of different values determinated above
        list_n_channels.append(n_channels)
        list_sample_freg.append(sample_freq)
        list_n_samples.append(n_samples)
        list_t_audio.append(t_audio)
        # list_signal_wave.append(signal_wave)
        list_signal_array.append(signal_array)
        list_times.append(times)


    #Convert all lists into a column of DataFrame
    df_raw['n_channel'] = list_n_channels
    df_raw['sample_freq'] = list_sample_freg
    df_raw['n_samples'] = list_n_samples
    df_raw['t_audio'] = list_t_audio
    # df_raw['signal_wave'] = list_signal_wave
    df_raw['signal_array'] = list_signal_array
    df_raw['times'] = list_times


    df_eda = df_raw

    return df_eda

##########################################################################################################

def cleaned_df(dataset: str) -> pd.DataFrame:

    """ This function takes the train.csv file (in the form of a dataframe) as argument.
    The path must may be adapted to absolute path
    The for loop iterate over all rows of the DataFrame train.csv :
    - The way is created by joining the initiale path(TBM) and the path of each files which is the column
    'file_path' of the train.csv DataFrame.
    - Calculation of each paramaters to plot the Signal Amplitude and the Frequency Spectrum
    - Plot
    """
    assert dataset == 'train' or dataset == 'test'

    # Paths endings for the initial train/test.csv files, and the cleaned train/test.csv files
    train_in = 'train.csv'
    test_in = 'test_full.csv'
    train_out = 'cleaned_train.csv'
    test_out = 'cleaned_test.csv'

    # Absolute paths to the local repository and .csv files
    path = os.path.abspath(os.path.dirname(os.getcwd()))

    train_csv_path = os.path.join(path,'raw_data', train_in)
    test_csv_path = os.path.join(path,'raw_data', test_in)

    cleaned_train_csv_path = os.path.join(path,'raw_data', train_out)
    cleaned_test_csv_path = os.path.join(path,'raw_data', test_out)

    # Train/test.csv paths to test if is.file() exists
    csv_train_path = Path(cleaned_train_csv_path)
    csv_test_path = Path(cleaned_test_csv_path)

    #Initiate empty lists
    list_n_samples =[]
    list_t_audio =[]
    list_signal_array=[]
    
    sample_rate = 16000

    if dataset == 'train':

        if csv_train_path.is_file():
            df_cleaned = load_cleaned_df(cleaned_train_csv_path)

        else:
            df_raw = pd.read_csv(train_csv_path)
            i = 0
            tot = len(df_raw)
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
                    # Extracting features from the audiofiles (optional, uncomment if needed)
                    # features=get_MFCC_features(audio, sample_rate)

                # append respective lists of different values determinated above
                list_n_samples.append(n_samples)
                list_t_audio.append(t_audio)
                list_signal_array.append(signal_array)
                # list_extracted_features.append(features)

            # Convert all lists into a column of DataFrame
            df_raw['n_samples'] = list_n_samples
            df_raw['t_audio'] = list_t_audio
            df_raw['signal_array'] = list_signal_array
            # df_raw['MFCC_features'] = list_extracted_features

            # Create an 'Amplitude' column
            mins = df_raw["signal_array"].apply(np.min)
            maxs = df_raw["signal_array"].apply(np.max)
            df_raw['amplitude'] = np.abs(maxs - mins)
            
            ohe = OneHotEncoder(sparse = False)
            # Fit encoder
            ohe.fit(df_raw[['speaker']])
            # Transform the current "Street" column
            df_raw[ohe.get_feature_names_out()] = ohe.transform(df_raw[['speaker']])

            # Dropping unnecessary columns
            df_cleaned = df_raw.drop(columns=['id', 'signal_array'])
            df_cleaned.reset_index(drop = True)

            
        # Save the file locally for future usage
        df_cleaned.to_csv(cleaned_train_csv_path)

    else:

        if csv_test_path.is_file():
            df_cleaned = load_cleaned_df(cleaned_test_csv_path)

        else:
            df_raw = pd.read_csv(test_csv_path)
            i = 0
            tot = len(df_raw)
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
                    # features=get_MFCC_features(audio, sample_rate)

                # append respective lists of different values determinated above
                list_n_samples.append(n_samples)
                list_t_audio.append(t_audio)
                list_signal_array.append(signal_array)
                # list_extracted_features.append(features)

            # Convert all lists into a column of DataFrame
            df_raw['n_samples'] = list_n_samples
            df_raw['t_audio'] = list_t_audio
            df_raw['signal_array'] = list_signal_array
            # df_raw['MFCC_features'] = list_extracted_features

            # Create an 'Amplitude' column
            mins = df_raw["signal_array"].apply(np.min)
            maxs = df_raw["signal_array"].apply(np.max)
            df_raw['amplitude'] = np.abs(maxs - mins)

            ohe = OneHotEncoder(sparse = False)
            # Fit encoder
            ohe.fit(df_raw[['speaker']])
            # Transform the current "Street" column
            df_raw[ohe.get_feature_names_out()] = ohe.transform(df_raw[['speaker']])
            
            # Dropping unnecessary columns
            df_cleaned = df_raw.drop(columns=['id', 'signal_array'])
            df_cleaned.reset_index(drop = True)

        # Save the file locally for future usage
        df_cleaned.to_csv(cleaned_test_csv_path)


    return df_cleaned
