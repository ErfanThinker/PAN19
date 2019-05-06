from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
from keras.utils import plot_model
from keras import layers, Model, optimizers
from keras.layers.merge import concatenate


def load_saved_model(model_file_path):
    return load_model(model_file_path)


# load models from file
def load_all_models(model_names_list):
    all_models = list()
    for model_name in model_names_list:
        # define filename for this ensemble
        #         filename = 'models/model_' + str(i + 1) + '.h5'
        filename = model_name + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# define stacked model from multiple member input models
def define_stacked_model(members, output_dim):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    #     print(ensemble_visible)
    #     ensemble_visible = [[ngram_input_tensor, word_input_tensor], convnet_input_tensor]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    #     ensemble_outputs = [concatenated, answer]
    merge = concatenate(ensemble_outputs)
    hidden = layers.Dense(128, activation='relu')(merge)
    hidden = layers.Dropout(0.3)(hidden)
    output = layers.Dense(output_dim, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    #     plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=3e-4), metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy, valX, valy, callback_list, batch_size, epochs=300, verbose=0):
    # prepare input data
    #     X = [inputX for _ in range(len(model.input))]
    # encode output data
    #     inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(inputX, inputy, validation_data=(valX, valy), batch_size=batch_size,
              callbacks=callback_list, epochs=epochs, verbose=verbose)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    #     X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(inputX, verbose=0)
