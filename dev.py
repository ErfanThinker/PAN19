# coding=utf-8
import json
import os

from keras import Model
from keras.layers import Dense, concatenate
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack

from MyUtils import clean_folder, read_files
from Word2Dim import Word2Dim


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'models/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


def freeze_layers(models):
    # update all layers in all models to not be trainable
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name


# define stacked model from multiple member input models
def define_stacked_model(members):
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
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(3, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(X, inputy_enc, epochs=300, verbose=0)


def do_keras_stuff(train_tokenized_indexed, test_tokenized_indexed, train_labels, test_labels, w2d, embedding_dim,
                   maxlen):
    from keras.layers import Embedding, Flatten, Dense
    from keras.models import Sequential
    from keras.optimizers import RMSprop
    from keras.utils import to_categorical
    from keras_preprocessing.sequence import pad_sequences
    import matplotlib.pyplot as plt
    # from sklearn.model_selection import train_test_split

    train_data = pad_sequences(train_tokenized_indexed, maxlen=maxlen)

    test_data = pad_sequences(test_tokenized_indexed, maxlen=maxlen)

    X_train, X_val, y_train, y_val = train_data, test_data, to_categorical(train_labels), to_categorical(test_labels)
    # X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels,
    #                                                   test_size=0.28, random_state=2019,
    #                                                   stratify=train_labels)

    # y_train = to_categorical(y_train)
    # y_val = to_categorical(y_val)

    model = Sequential()
    model.add(Embedding(w2d.word_embedding.shape[0], embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(set(train_labels)), activation='softmax'))
    model.summary()

    # model.layers[0].set_weights([w2d.word_embedding])
    # model.layers[0].trainable = False

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=120,
                        batch_size=1)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def main():
    dataset_path = '.' + os.sep + 'pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    outpath = '.' + os.sep + 'dev_out'

    clean_folder(outpath)

    infocollection = dataset_path + os.sep + 'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    for index, problem in enumerate(problems):
        infoproblem = dataset_path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])

        candidates.sort()
        # Building training set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(dataset_path + os.sep + problem, candidate))
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        index_2_label_dict = {i: l for i, l in enumerate(set(train_labels))}
        label_2_index_dict = {l: i for i, l in enumerate(set(train_labels))}
        train_labels = [label_2_index_dict[v] for v in train_labels]
        w2d = Word2Dim()
        train_tokenized_with_pos, train_tokenized_indexed = w2d.fit_transform_texts(train_texts, train_labels,
                                                                                    language[index])

        maxlen = len(max(train_tokenized_indexed, key=len))  # We will cut the texts after # words
        embedding_dim = w2d.word_embedding.shape[1]

        # preparing test set
        ground_truth_file = dataset_path + os.sep + problem + os.sep + 'ground-truth.json'
        gt = {}
        with open(ground_truth_file, 'r') as f:
            for attrib in json.load(f)['ground_truth']:
                gt[attrib['unknown-text']] = attrib['true-author']

        test_docs = read_files(dataset_path + os.sep + problem, unk_folder, gt)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_labels = [label for i, (text, label) in enumerate(test_docs)]

        # Filter validation to known authors
        test_texts = [text for i, (text, label) in enumerate(test_docs) if label in label_2_index_dict.keys()]
        test_labels = [label for i, (text, label) in enumerate(test_docs) if label in label_2_index_dict.keys()]

        test_labels = [label_2_index_dict[v] for v in test_labels]

        test_tokenized_with_pos, test_tokenized_indexed = w2d.transform(test_texts)

        do_keras_stuff(train_tokenized_indexed, test_tokenized_indexed, train_labels, test_labels, w2d, embedding_dim,
                       maxlen)


if __name__ == '__main__':
    main()
