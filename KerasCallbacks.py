from keras import callbacks

callbacks_list_neu = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
    ),
    callbacks.ModelCheckpoint(
        filepath='my_model_neu.h5',
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
        filepath='my_model_neu_ngrams.h5',
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
        filepath='my_model_neu_words.h5',
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
        filepath='my_model_stacked.h5',
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
