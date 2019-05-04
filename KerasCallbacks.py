from keras import callbacks
import os

models_folder = '.' + os.sep + 'ms_models'
callbacks_list_neu_path = models_folder + os.sep + 'my_model_neu'
callbacks_list_neu_ngrams_path = models_folder + os.sep + 'my_model_neu_ngrams'
callbacks_list_neu_words_path = models_folder + os.sep + 'my_model_neu_words'
callbacks_list_stacked_path = models_folder + os.sep + 'my_model_stacked'

callbacks_list_neu = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
    ),
    callbacks.ModelCheckpoint(
        filepath=callbacks_list_neu_path + '.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=20,
    )
]

callbacks_list_neu_ngrams = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
    ),
    callbacks.ModelCheckpoint(
        filepath=callbacks_list_neu_ngrams_path + '.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=20,
    )
]

callbacks_list_neu_words = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
    ),
    callbacks.ModelCheckpoint(
        filepath=callbacks_list_neu_words_path + '.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=20,
    )
]

callbacks_list_convnet = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=35,
    ),
    callbacks.ModelCheckpoint(
        filepath='my_model_convnet.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=10,
    )
]

callbacks_list_stacked = [
    callbacks.ModelCheckpoint(
        filepath=callbacks_list_stacked_path + '.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=20,
    )
]
