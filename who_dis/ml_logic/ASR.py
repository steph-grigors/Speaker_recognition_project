import pandas as pd
import numpy as np
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
ASRmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def get_transcript(signal_array):
    input_as_list = [float(x) for x in signal_array]
    # tokenize
    input_values = processor(input_as_list, return_tensors="pt", padding="longest").input_values

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

def get_audio_array(audiofile):
    wav_obj = wave.open(audiofile, 'rb')
    n_samples = wav_obj.getnframes()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    return signal_array


if __name__ == '__main__':
    pass
