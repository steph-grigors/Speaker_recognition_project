from colorama import Fore, Style
from typing import Tuple
import numpy as np
from tensorflow import keras
from keras import Model, regularizers, optimizers, callbacks
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.metrics import Recall, Precision
from who_dis.params import *


def init_baseCNN():
    '''
    This function takes has no parameters. It returns an uncompiled baseline CNN model that has the following structure:
    - 1 Conv2D layer with 5 channels, kernal shape (3,3), activation "relu"
    - 1 Flatten layer
    - 1 Dense output layer with 18 neurons, activation "softmax"
    '''
    reg_l1 = regularizers.L1(0.01)
    reg_l2 = regularizers.L2(0.01)
    reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.0005)

    model = Sequential()
    model.add(Conv2D(4, (3,3), activation='relu', input_shape=input_shape, activity_regularizer=reg_l1_l2))
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
    model.add(Dense(40,activation='relu',input_dim=40))
    model.add(Dense(18,activation='softmax'))

    print("✅ model ANN initialized")

    return model

def basic_compiler(model: Model,
                   learning_rate=learning_rate) -> Model:
    '''
    This function takes in a model, compiles it and returns the compiled model.
    Compiler parameters:
    - optimizer: "adam"
    - loss: "categorical_crossentropy"
    - metrics: [tensorflow.metrics.Recall(),
                tensorflow.metrics.Precision()]
    '''
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                                initial_learning_rate=learning_rate,
                                                decay_steps=10000,
                                                decay_rate=0.9)

    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', Recall(), Precision()])

    print("✅ model compiled")

    return model

def grid_search(model: Model,
                X_train,
                y_train):

    # Hyperparameter Grid
    grid = {
        'l1_ratio': [0.2, 0.5, 0.8],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64, 128],
        'epochs': [5, 10, 20]
        }

    # Instantiate Grid Search
    search = GridSearchCV(
        model,
        grid,
        scoring = 'accuracy',
        cv = 10,
        n_jobs=-1
    )

    # Fit data to Grid Search
    search.fit(X_train, y_train);

    return search.best_params_


def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                batch_size=batch_size,
                patience=patience,
                validation_split=validation_split) -> Tuple[Model, dict]:
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
