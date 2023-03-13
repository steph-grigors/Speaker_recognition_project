import numpy as np
import pandas as pd



def preprocess() -> None:

    from who_dis.ml_logic.data import cleaned_df
    from who_dis.ml_logic.preprocess import MFCC_features_extractor, MEL_spect_features_extractor, OHE_target

    # Process data
    data_clean = cleaned_df(dataset='train')
    X1 = MFCC_features_extractor(dataset='train')
    X2 = MEL_spect_features_extractor(dataset='train')
    y1 = OHE_target(data_clean)
    y2 = OHE_target(data_clean)

    print("✅ preprocess() done \n")

def train():

    from who_dis.ml_logic.registry import load_model, save_model, save_results
    from who_dis.ml_logic.model import init_baseCNN, init_baseCNN, basic_compiler, train_model, evaluate_model

    # Train model using `model.py`
    model = load_model()
    if model is None:
        model = init_baseCNN()
        model = basic_compiler(model, learning_rate=0.001)

    model, history = train_model(model, X_train, y_train,
                                batch_size=64,
                                patience=5,
                                validation_split=0.2)

    val_accuracy = np.min(history.history['accuracy'])
    val_precision = np.min(history.history['precision'])
    val_recall = np.min(history.history['recall'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")
    return val_mae

def evaluate():
    """
    Evaluate the performance of the latest production model on processed data
    Returns accuracy, recall, precision
    """

    from who_dis.ml_logic.registry import load_model

    model = load_model(stage=stage)
    assert model is not None

    pass

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from who_dis.ml_logic.registry import load_model

    print("\n⭐️ Use case: predict")

    model = load_model()
    assert model is not None

    breakpoint()
    ######## NEED TO CREATE A PREPROCESS PIPELINE FUNCT###
    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred
