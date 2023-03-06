Speaker Recognition - CMU ARCTIC

https://www.kaggle.com/datasets/mrgabrielblins/speaker-recognition-cmu-arctic

File information
train.csv - file containing all the data you need for training, with 4 columns, id (file id), file_path(path to .wav files), speech(transcription of audio file), and speaker (target column)
test.csv - file containing all the data you need to test your model (20% of total audio files), it has the same columns as train.csv
train/ - Folder with training data, subdivided with Speaker's folders
aew/ - Folder containing audio files in .wav format for speaker aew
…
test/ - Folder containing audio files for test data.

Column description
Column	Description
id	file id (string)
file_path	file path to .wav file (string)
speech	transcription of the audio file (string)
speaker	speaker name, use this as the target variable if you are doing audio classification (string)
