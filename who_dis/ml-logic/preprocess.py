import wave
import os
import numpy as np

def cleaned_df(df_raw):

    """ This function takes the train.csv file as argument.
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

    for row in range(len(df_raw)):
        wav_obj = wave.open(os.path.join(path,'raw_data',df_raw['file_path'][row]), 'rb')
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
        # Turn signal wave into numpy arry to get signal values from this
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        # Need to calculate the time at which each sample is taken before plotting signal values
        times = np.linspace(0, n_samples/sample_freq, num=n_samples)

        # append respective lists of different values determinated above
        list_n_channels.append(n_channels)
        list_sample_freg.append(sample_freq)
        list_n_samples.append(n_samples)
        list_t_audio.append(t_audio)
        list_signal_wave.append(signal_wave)
        list_signal_array.append(signal_array)
        list_times.append(times)

    #Convert all lists into a column of DataFrame
    df_raw['n_channel'] = list_n_channels
    df_raw['sample_freq'] = list_sample_freg
    df_raw['n_samples'] = list_n_samples
    df_raw['t_audio'] = list_t_audio
    df_raw['signal_wave'] = list_signal_wave
    df_raw['signal_array'] = list_signal_array
    df_raw['times'] = list_times

    df_cleaned = df_raw

    return df_cleaned
