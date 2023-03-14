import numpy as np
import pandas as pd
from who_dis.params import *
from sklearn.preprocessing import OneHotEncoder
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

    # save_preprocessed(X1_train, X_train, y_train, 'train')
    # save_preprocessed(X1_test, X_test, y_test, 'test')

    print("✅ preprocess() done \n")

    return cleaned_train, cleaned_test, X_train, X_test, y_train, y_test


def train(X_train, y_train):

    # Train model using `model.py`
    model = None
    # load_model(stage="Production")
    if model is None:
        model = init_baseCNN()
        model = basic_compiler(model, learning_rate=learning_rate)

    model, history = train_model(model, X_train, y_train,
                                batch_size=batch_size,
                                patience=patience,
                                validation_split=validation_split)

    val_accuracy = np.min(history.history['accuracy'])
    # val_precision = np.min(history.history['precision'])
    # val_recall = np.min(history.history['recall'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy = val_accuracy,
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
    metrics = evaluate_model(model, X_test, y_test, batch_size=batch_size)

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

    audio_pred, sample_rate_pred = load_audio_file(audiofile)
    X_pred = get_MEL_spectrogram(audio_pred, sample_rate_pred)

    X_pred_df = load_preprocessed('test')
    columns_names = X_pred_df.columns.tolist()
    unwanted_columns_names = ['file_path',
                        'speech',
                        'speaker',
                        'n_samples',
                        't_audio',
                        'amplitude']
    classes = [name for name in columns_names if name not in unwanted_columns_names]
    y_pred = model.predict(X_pred)
    name_pred = {classes[np.argmax(y_pred)]}

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    print(f"\n✅ The person whom voice you heard is: {classes[np.argmax(y_pred)]}" "\n")

    return name_pred, y_pred

if __name__ == "__main__":
    pass
