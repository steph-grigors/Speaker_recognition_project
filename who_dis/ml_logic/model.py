from colorama import Fore, Style
from typing import Tuple
import numpy as np
from tensorflow import keras
from keras import Model, regularizers, optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.metrics import Recall, Precision


def init_baseCNN():
    '''
    This function takes has no parameters. It returns an uncompiled baseline CNN model that has the following structure:
    - 1 Conv2D layer with 5 channels, kernal shape (3,3), activation "relu"
    - 1 Flatten layer
    - 1 Dense output layer with 18 neurons, activation "softmax"
    '''
    model = Sequential()
    model.add(Conv2D(5, (3,3), activation='relu', input_shape=(128, 751, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(18, activation='softmax'))

    print("✅ model CNN initialized")

    return model

def init_baseNN():
    '''
    This function takes has no parameters. It returns an uncompiled baseline NN model that has the following structure:
    - 1 Dense layer with 50 neurons, "relu" activation, and input_dim=40
    - 1 Dense output layer with 18 neurons, activation "softmax"
    '''
    model = Sequential()
    model.add(Dense(50,activation='relu',input_dim=40))
    model.add(Dense(18,activation='softmax'))

    print("✅ model ANN initialized")

    return model

def basic_compiler(model: Model, learning_rate=0.001) -> Model:
    '''
    This function takes in a model, compiles it and returns the compiled model.
    Compiler parameters:
    - optimizer: "adam"
    - loss: "categorical_crossentropy"
    - metrics: [tensorflow.metrics.Recall(),
                tensorflow.metrics.Precision()]
    '''
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', Recall(), Precision()])

    print("✅ model compiled")

    return model

def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                batch_size=64,
                patience=5,
                validation_split=0.2) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = callbacks.EarlyStopping(monitor="val_loss",
                                 patience=patience,
                                 restore_best_weights=True)

    history = model.fit(X_train, y_train,
          batch_size=batch_size, # Batch size -too small--> no generalization
          epochs=5,    #            -too large--> slow computations
          validation_split=validation_split,
          callbacks=[es],
          verbose=1)

    print(f"✅ model trained on {len(X_train)} rows")

    return model, history


def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   batch_size=int) -> Tuple[Model, dict]:

    """
    Evaluate trained model performance on dataset
    """
    print(Fore.BLUE + f"\nEvaluate model on {len(X_test)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        verbose=1,
        return_dict=True
        )

    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']

    print(f"✅ model evaluated: accuracy, precision, recall {round(accuracy, 3)}, {round(precision, 3)}, {round(recall, 3)}")

    return metrics

def predict_model(model, X_test):

   model_prediction = model.predict(X_test)

   return model_prediction
