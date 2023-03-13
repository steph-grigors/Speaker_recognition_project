import numpy as np
import pandas as pd
from who_dis.params import *
from who_dis.ml_logic.registry import save_preprocessed



def preprocess() -> None:

    from who_dis.ml_logic.data import cleaned_df
    from who_dis.ml_logic.preprocess import MFCC_features_extractor, MEL_spect_features_extractor, OHE_target

    # Process data
    cleaned_train = cleaned_df(dataset='train')
    cleaned_test = cleaned_df(dataset='test')

    X1_train = MFCC_features_extractor(dataset='train')
    X1_test = MFCC_features_extractor(dataset='test')
    X_train = MEL_spect_features_extractor(dataset='train')
    X_test = MEL_spect_features_extractor(dataset='test')
    y_train = OHE_target(cleaned_train)
    y_test = OHE_target(cleaned_test)

    save_preprocessed(X1_train, X_train, y_train)
    save_preprocessed(X1_test, X_test, y_test)

    print("✅ preprocess() done \n")


def train(X_train, y_train):

    from who_dis.ml_logic.registry import load_model, save_model, save_results
    from who_dis.ml_logic.model import init_baseCNN, init_baseCNN, basic_compiler, train_model

    # Train model using `model.py`
    model = load_model()
    if model is None:
        model = init_baseCNN()
        model = basic_compiler(model, learning_rate=learning_rate)

    model, history = train_model(model, X_train, y_train,
                                batch_size=batch_size,
                                patience=patience,
                                validation_split=validation_split)

    val_accuracy = np.min(history.history['accuracy'])
    val_precision = np.min(history.history['precision'])
    val_recall = np.min(history.history['recall'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy = val_accuracy,
                                             precision = val_precision,
                                             recall = val_recall))

    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_accuracy


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


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from who_dis.ml_logic.registry import load_model

    print("\n⭐️ Use case: predict")

    model = load_model()
    assert model is not None

    X_pred = load_preprocessed(X_pred)
    y_pred = model.predict(X_pred)

    ######## NEED TO CREATE A PREPROCESS PIPELINE FUNCT###
    # X_processed = preprocess_features(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred
