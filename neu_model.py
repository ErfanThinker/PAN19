from numpy import argmax
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(1)

from keras import layers, regularizers
from keras.models import Sequential, load_model
from keras import optimizers


def build_model(data, labels):
    neu_ml = Sequential()
    neu_ml.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                            input_shape=(data.shape[1],)))
    neu_ml.add(layers.Dropout(0.3))
    neu_ml.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    neu_ml.add(layers.Dropout(0.3))
    neu_ml.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    neu_ml.add(layers.Dropout(0.3))
    neu_ml.add(layers.Dense(len(set(labels)), activation='softmax'))

    neu_ml.compile(optimizer=optimizers.Adam(lr=1e-4),
                   loss='categorical_crossentropy',
                   metrics=['acc'])
    # neu_ng.summary()

    return neu_ml


def fit(model, train_data, val_data, train_labels, val_labels, batch_size, callbacks, verbose=1, epochs=500):
    model.fit(train_data, train_labels,
              validation_data=(val_data, val_labels),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose=verbose
              )


def load(model_name):
    return load_model(model_name + '.h5')


def predict(model, test_data, return_labels=False):
    yhat = model.predict(test_data)
    return yhat if not return_labels else argmax(yhat, axis=1)
