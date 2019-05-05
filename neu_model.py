from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(1)
from keras import layers, regularizers
from keras.models import Sequential, load_model
from keras import optimizers
from numpy import argmax


def build(dim, num_of_classes, capacities, dropout):
    neu_ml = Sequential()

    neu_ml.add(layers.Dense(capacities[0], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                            input_shape=(dim,)))
    neu_ml.add(layers.Dropout(dropout))

    for index, capacity in enumerate(capacities):
        if index > 0:
            neu_ml.add(
                layers.Dense(capacity, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
            neu_ml.add(layers.Dropout(dropout))

    neu_ml.add(layers.Dense(num_of_classes, activation='softmax'))

    neu_ml.compile(optimizer=optimizers.Adam(lr=1e-4),
                   loss='categorical_crossentropy',
                   metrics=['acc'])
    # neu_ng.summary()

    return neu_ml


def fit_model(model, train_data, val_data, train_labels, val_labels, batch_size, callbacks, verbose=1, epochs=100):
    model.fit(train_data, train_labels,
              validation_data=(val_data, val_labels),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose=verbose
              )


def load(model_name):
    return load_model(model_name + '.h5')


def predict_data(model, test_data, return_labels=False):
    yhat = model.predict(test_data)
    return yhat if not return_labels else argmax(yhat, axis=1)
