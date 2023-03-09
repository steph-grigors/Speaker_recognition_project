from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.metrics import Recall, Precision

def init_baseCNN():
    '''
    This function takes has no parameters. It returns an uncompiled baseline CNN model that has the following structure:
    - 1 Conv2D layer with 5 channels, kernal shape (3,3), activation "relu"
    - 1 Flatten layer
    - 1 Dense output layer with 18 neurons, activation "softmax"
    '''
    model = Sequential()
    model.add(Conv2D(5, (3,3), activation='relu', input_shape=(600, 257,1)))
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
                 metrics=[Recall(), Precision()])
    return model
