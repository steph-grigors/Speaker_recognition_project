from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import callbacks


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
    model.add(Dropout(rate=0.4))
    model.add(Flatten())
    model.add(Dense(18, activation='softmax'))

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

    return model

def basic_compiler(model):
    '''
    This function takes in a model, compiles it and returns the compiled model.
    Compiler parameters:
    - optimizer: "adam"
    - loss: "categorical_crossentropy"
    - metrics: [tensorflow.metrics.Recall(),
                tensorflow.metrics.Precision()]
    '''
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy', Recall(), Precision()])

    return model

def train_model(model, X_train, y_train):

    es = callbacks.EarlyStopping(patience=20, restore_best_weights=True)

    history = model.fit(X_train, y_train,
          batch_size=64, # Batch size -too small--> no generalization
          epochs=5,    #            -too large--> slow computations
          validation_split=0.2,
          callbacks=[es],
          verbose=1)

    return history


def evaluate_model(model, X_test, y_test):

 model_evaluate = model.evaluate(X_test, y_test, verbose=0)

 return model_evaluate

def predict_model(model, X_test):

   model_prediction = model.predict(X_test)

   return model_prediction
