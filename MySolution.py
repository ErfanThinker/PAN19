# coding=utf-8
from __future__ import division
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import multiprocessing
from MyUtils import extract_n_grams

from MyUtils import clean_folder, read_files, shuffle_docs, shuffle_docs2
from Word2Dim import Word2Dim
import argparse
import codecs
import glob
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from numpy import argmax
from sklearn.metrics import accuracy_score
from keras import layers, Input
from keras.models import Sequential, Model, load_model
from keras import optimizers, regularizers
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from MyUtils import clean_folder



def process_problem(problem, path, n, tf, language, problem_index, pt, outpath):
    infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
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
        train_docs.extend(read_files(path + os.sep + problem, candidate))
    train_texts = [text for i, (text, label) in enumerate(train_docs)]
    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    initial_train_size = len(train_labels)

    # this will produce combination of train docs for validation purpose
    train_texts, train_labels = shuffle_docs(train_texts, train_labels)
    validation_size = len(train_texts) - initial_train_size
    class_size = int(initial_train_size / len(set(train_labels)))

    # train_texts, train_labels, validation_start_index, class_size = shuffle_docs2(train_texts, train_labels)

    index_2_label_dict = {i: l for i, l in enumerate(set(train_labels))}
    label_2_index_dict = {l: i for i, l in enumerate(set(train_labels))}
    train_labels = [label_2_index_dict[v] for v in train_labels]

    w2d = Word2Dim(lang=language[problem_index])
    train_tokenized_with_pos, train_tokenized_indexed = w2d.fit_transform_texts(train_texts, train_labels, tf=tf)

    # building test set
    test_docs = read_files(path + os.sep + problem, unk_folder)
    test_texts = [text for i, (text, label) in enumerate(test_docs)]
    test_tokenized_with_pos, test_tokenized_indexed = w2d.transform(test_texts)


    maxlen = len(max(train_tokenized_indexed, key=len))  # We will cut the texts after # words
    embedding_dim = w2d.word_embedding.shape[1]

    vocabulary = extract_n_grams(train_docs, n, tf)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False, vocabulary=vocabulary)

    n_gram_train_data = vectorizer.fit_transform(train_texts)
    n_gram_train_data = n_gram_train_data.astype(float)
    for i, v in enumerate(train_texts):
        n_gram_train_data[i] = n_gram_train_data[i] / len(train_texts[i])

    n_gram_test_data = vectorizer.transform(test_texts)
    n_gram_test_data = n_gram_test_data.astype(float)
    for i, v in enumerate(test_texts):
        n_gram_test_data[i] = n_gram_test_data[i] / len(test_texts[i])

    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data_ngrams = max_abs_scaler.fit_transform(n_gram_train_data)
    scaled_test_data_ngrams = max_abs_scaler.transform(n_gram_test_data)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data_words = max_abs_scaler.fit_transform(
        w2d.get_texts_vectorized_and_normalized(train_tokenized_indexed)[:, 1:])
    scaled_test_data_words = max_abs_scaler.transform(
        w2d.get_texts_vectorized_and_normalized(test_tokenized_indexed)[:, 1:])

    train_data = pad_sequences(train_tokenized_indexed, maxlen=maxlen)

    test_data = pad_sequences(test_tokenized_indexed, maxlen=maxlen)

    train_val_split_index = initial_train_size
    y_train, y_val = train_labels[:train_val_split_index], train_labels[train_val_split_index:]
    X_train, X_val = train_data[:train_val_split_index], train_data[train_val_split_index:]
    X_scaled_train_data_words, X_scaled_val_data_words = scaled_train_data_words[
                                                         :train_val_split_index], scaled_train_data_words[
                                                                                  train_val_split_index:]
    X_scaled_train_data_ngrams, X_scaled_val_data_ngrams = scaled_train_data_ngrams[
                                                           :train_val_split_index], scaled_train_data_ngrams[
                                                                                    train_val_split_index:]

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)






    # Reject option (used in open-set cases)
    # count = 0
    # for i, p in enumerate(predictions):
    #     sproba = sorted(proba[i], reverse=True)
    #     if sproba[0] - sproba[1] < pt:
    #         predictions[i] = u'<UNK>'
    #         count = count + 1
    # print('\t', count, 'texts left unattributed')
    # # Saving output data
    # out_data = []
    # unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
    # pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
    # for i, v in enumerate(predictions):
    #     out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
    # with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
    #     json.dump(out_data, f, indent=4)
    # print('\t', 'answers saved to file', 'answers-' + problem + '.json')


def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()
    clean_folder(outpath)
    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    for problem_index, problem in enumerate(problems):
        process_problem(problem, path, n, ft, language, problem_index, pt, outpath)

    print('elapsed time:', time.time() - start_time)


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-19 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-pt', type=float, default=0.1, help='probability threshold for the reject option (default=0.1')
    args = parser.parse_args()
    if not args.i:
        args.i = '.\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
        # print('ERROR: The input folder is required')
        # parser.exit(1)
    if not args.o:
        args.o = '.\\ms_out'
        # print('ERROR: The output folder is required')
        # parser.exit(1)

    baseline(args.i, args.o, args.n, args.ft, args.pt)


if __name__ == '__main__':
    main()
