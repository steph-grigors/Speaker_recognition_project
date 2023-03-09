import wave
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tqdm

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

    #Initiate empty lists
    list_n_samples =[]
    list_t_audio =[]
    list_signal_array=[]
    list_extracted_features = []

    #Sample_frequency
    sample_rate = 16000

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

        # Extracting features from the audiofiles
        features=base_MFCC_features_extractor(filename)[2]

        # append respective lists of different values determinated above
        list_n_samples.append(n_samples)
        list_t_audio.append(t_audio)
        list_signal_array.append(signal_array)
        list_extracted_features.append(features)

    #Convert all lists into a column of DataFrame
    df_raw['n_samples'] = list_n_samples
    df_raw['t_audio'] = list_t_audio
    df_raw['signal_array'] = list_signal_array
    df_raw['MFCC_features'] = list_extracted_features

    #Create an 'Amplitude' column
    mins = df_raw["signal_array"].apply(np.min)
    maxs = df_raw["signal_array"].apply(np.max)
    df_raw['amplitude'] = np.abs(maxs - mins)

    df_cleaned = df_raw.drop(columns=['id', 'signal_array'])

    return df_cleaned

def clipping_unbalanced_classes(df_cleaned):
    '''Checks for the number of observations per target class and clips
    the audio files in 2 clips if values_count of that class < max(values_count))'''

    for num_recordings in df_cleaned.speaker.value_counts(normalize = True):
        if num_recordings > df_cleaned.speaker.value_counts().max() // 2:
            pass #clip the audio file in 2 windows'''

def trim_recordings():
    pass

def base_MFCC_features_extractor(filename):
    '''
    audio represents the n_samples taken with a 16Khz frequency rate
    sample_rate is set to None as to use the native sampling rate
    mono = True sets the n_channels to 1
    mfccs_scaled_features is an array of shape (40, ) MFC coefficients that represent our features
    '''

    #load the file (audio)
    audio, sample_rate = librosa.load(filename, sr= None, mono = True, res_type='soxr_hq')
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return audio, sample_rate, np.array(mfccs_scaled_features)

def show_MEL_spectrogram(audio, sample_rate):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=16000, x_axis='time');
    plt.title('Mel Spectrogram');
    plt.colorbar(format='%+2.0f dB');

    return mel_spect

def get_MEL_spectrogram(audio, sample_rate):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return mel_spect


##########################################################################################################

# def final_MFCC_features_extractor(df_raw):

#     path = os.path.dirname(os.getcwd())

#     extracted_features=[]
#     i = 0
#     tot = len(df_raw)

#     for row in range(len(df_raw)):
#         i += 1
#         print(f"Treating {i} / {tot}", end='\r')
#         filename = os.path.join(path,'raw_data',df_raw['file_path'][row])
#         # class_labels=row["speaker"]
#         features=base_MFCC_features_extractor(filename)
#         extracted_features.append(features)

#     return np.array(extracted_features)

###########################################################################################################
# def get_amplitude(df_cleaned):
#     '''Creates a new column called "Amplitude" by
#     computing the abs of the diff between max and min signal, then dropping the signal_array column'''
#     mins = df_cleaned["signal_array"].apply(np.min)
#     maxs = df_cleaned["signal_array"].apply(np.max)
#     df_cleaned['amplitude'] = np.abs(maxs - mins)
#     df_cleaned.drop(columns=['signal_array'], inplace = True)

#     return df_cleaned
