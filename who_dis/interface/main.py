import numpy as np
import pandas as pd
from who_dis.params import *
from sklearn.preprocessing import OneHotEncoder
from who_dis.ml_logic.data import cleaned_df
from who_dis.ml_logic.registry import save_preprocessed, load_preprocessed, load_model, save_model, save_results, load_audio_file
from who_dis.ml_logic.model import init_baseCNN, init_baseCNN, basic_compiler, train_model
from who_dis.ml_logic.preprocess import get_MEL_spectrogram



def preprocess() -> None:

    from who_dis.ml_logic.data import cleaned_df
    from who_dis.ml_logic.preprocess import MFCC_features_extractor, MEL_spect_features_extractor, OHE_target

    # Process data
    cleaned_train = cleaned_df(dataset='train')
    cleaned_test = cleaned_df(dataset='test')

    # X1_train = MFCC_features_extractor(cleaned_train, dataset='train')
    # X1_test = MFCC_features_extractor(cleaned_test, dataset='test')
    X_train = MEL_spect_features_extractor(cleaned_train, dataset='train')
    X_test = MEL_spect_features_extractor(cleaned_test, dataset='test')
    y_train = OHE_target(cleaned_train)
    y_test = OHE_target(cleaned_test)

    save_preprocessed(X_train, y_train, 'train')
    save_preprocessed(X_test, y_test, 'test')

    print("✅ preprocess() done \n")

    return cleaned_train, cleaned_test, X_train, X_test, y_train, y_test


def train(X_train, y_train):

    model = None
    # load_model()
    if model is None:
        model = init_baseCNN()
        model = basic_compiler(model, learning_rate=0.01)

    model, history = train_model(model, 
                                 X_train, 
                                 y_train,
                                batch_size=32,
                                epochs=50,
                                patience=10,
                                validation_split=0.2)

    val_categorical_accuracy = np.mean(history.history['val_categorical_accuracy'])
    val_precision = np.mean(history.history['val_precision'])
    val_recall = np.mean(history.history['val_recall'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy = val_categorical_accuracy,
                                             precision = val_recall,
                                             recall = val_recall
                                             ))

    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return history


def evaluate(X_test, y_test):
    """
    Evaluate the performance of the latest production model on processed data
    Returns accuracy, recall, precision
    """

    from who_dis.ml_logic.registry import load_model
    from who_dis.ml_logic.model import evaluate_model

    model = load_model()
    assert model is not None
    metrics = evaluate_model(model, X_test, y_test, batch_size=32)

    print("✅ evaluate() done \n")

    return metrics


def pred(audiofile):
    """
    Make a prediction using the latest trained model
    """
    from who_dis.ml_logic.registry import load_model

    print("\n⭐️ Use case: predict")

    model = load_model()
    assert model is not None
    
    # Preprocessing the audiofile
    audio_pred, sample_rate_pred = load_audio_file(audiofile)
    X_pred = get_MEL_spectrogram(audio_pred, sample_rate_pred)
    X_pred = X_pred.reshape((-1,128,606,1))

    speaker_names = {0: 'Andrew',
                     1: 'Maximilian',
                     2: 'Parul',
                     3: 'Mike',
                     4: 'Arya',
                     5: 'Henry',
                     6: 'Chloe',
                     7: 'Laura',
                     8: 'Samuel',
                     9: 'Krish',
                     10: 'Jim',
                     11: 'Alex',
                     12: 'Kalindi',
                     13: 'Elena',
                     14: 'Walter White',
                     15: 'Jules',
                     16: 'Pascaline',
                     17: 'Kamilla'}
                               
                     
    # Computing y_pred and the speaker's name
    y_pred = model.predict(X_pred)
    name_pred = speaker_names[np.argmax(y_pred)]

    print(f"\n✅ prediction done: {y_pred} \n")
    print(f"\n✅ The person whom voice you heard is: {name_pred} \n")

    return y_pred, name_pred

if __name__ == "__main__":
    pass
