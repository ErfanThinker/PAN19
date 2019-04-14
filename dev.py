# coding=utf-8
import json

import matplotlib.pyplot as plt
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from nltk import *
from sklearn.model_selection import train_test_split

from MyUtils import clean_folder, read_files
from Word2Dim import Word2Dim


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
        train_labels = sorted([label_2_index_dict[v] for v in train_labels])
        w2d = Word2Dim()
        train_tokenized_with_pos, train_tokenized_indexed = w2d.fit_transform_texts(train_texts, train_labels,
                                                                                    language[index])

        maxlen = len(max(train_tokenized_indexed, key=len))  # We will cut the texts after # words
        # max_words = 10000  # We will only consider the top 10,000 words in the dataset
        # tokenizer = Tokenizer(num_words=max_words)
        # tokenizer.fit_on_texts(train_texts)
        # sequences = tokenizer.texts_to_sequences(train_texts)
        train_data = pad_sequences(train_tokenized_indexed, maxlen=maxlen)

        embedding_dim = w2d.word_embedding.shape[1]

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels,
                                                          test_size=0.28, random_state=2019,
                                                          stratify=train_labels)

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        model = Sequential()
        model.add(Embedding(w2d.word_embedding.shape[0], embedding_dim, input_length=maxlen))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(set(train_labels)), activation='softmax'))
        model.summary()

        model.layers[0].set_weights([w2d.word_embedding])
        model.layers[0].trainable = False

        model.compile(optimizer='rmsprop',
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


if __name__ == '__main__':
    main()
