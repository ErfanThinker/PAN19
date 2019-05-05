# coding=utf-8
from __future__ import division

from numpy.random import seed

import KerasCallbacks as kc
from KerasUtils import pad_seqs, encode_labels

seed(1)
from tensorflow import set_random_seed

set_random_seed(1)
from MyUtils import extract_n_grams, read_files, shuffle_docs

from Word2Dim import Word2Dim
import argparse
import json
import os
import time
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from MyUtils import clean_folder
import neu_model as nm
import stacked_model as sm


def process_problem(problem, path, n, tf, language, problem_index, pt, outpath):
    print('Processing :', problem)
    infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
    candidates = []
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        unk_folder = fj['unknown-folder']
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    candidates.sort()

    clean_folder('.' + os.sep + 'ms_models')

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
    num_of_classes = len(set(train_labels))
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

    # train_data = pad_seqs(train_tokenized_indexed, maxlen=maxlen)

    # test_data = pad_seqs(test_tokenized_indexed, maxlen=maxlen)

    train_val_split_index = initial_train_size
    y_train, y_val = train_labels[:train_val_split_index], train_labels[train_val_split_index:]
    # X_train, X_val = train_data[:train_val_split_index], train_data[train_val_split_index:]
    X_scaled_train_data_words, X_scaled_val_data_words = scaled_train_data_words[
                                                         :train_val_split_index], scaled_train_data_words[
                                                                                  train_val_split_index:]
    X_scaled_train_data_ngrams, X_scaled_val_data_ngrams = scaled_train_data_ngrams[
                                                           :train_val_split_index], scaled_train_data_ngrams[
                                                                                    train_val_split_index:]

    y_train = encode_labels(y_train)
    y_val = encode_labels(y_val)

    ngram_model_capacity = [32, 64, 64]
    ngram_model = nm.build(X_scaled_train_data_ngrams.shape[1], num_of_classes, ngram_model_capacity, dropout=0.3)
    nm.fit_model(ngram_model, X_scaled_train_data_ngrams, X_scaled_val_data_ngrams, y_train, y_val,
                 batch_size=class_size, callbacks=kc.get_callbacks_list_neu_ngrams(), verbose=0)

    words_model_capacity = [32, 64, 64]
    words_model = nm.build(X_scaled_train_data_words.shape[1], num_of_classes, words_model_capacity, dropout=0.3)
    nm.fit_model(words_model, X_scaled_train_data_words, X_scaled_val_data_words, y_train, y_val,
                 batch_size=class_size, callbacks=kc.get_callbacks_list_neu_words(), verbose=0)

    members = sm.load_all_models([kc.callbacks_list_neu_ngrams_path, kc.callbacks_list_neu_words_path])
    # members = [ngram_model, words_model]
    # define ensemble model
    stacked_model = sm.define_stacked_model(members, num_of_classes)

    # fit stacked model on test dataset
    sm.fit_stacked_model(stacked_model, [X_scaled_train_data_ngrams, X_scaled_train_data_words], y_train,
                         [X_scaled_val_data_ngrams, X_scaled_val_data_words], y_val,
                         callback_list=kc.get_callbacks_list_stacked(), batch_size=class_size)
    final_model = sm.load_saved_model(kc.callbacks_list_stacked_path + '.h5')
    # make predictions and evaluate
    yhat = sm.predict_stacked_model(final_model, [scaled_test_data_ngrams, scaled_test_data_words])

    np.save('.' + os.sep + 'ms_out_raw' + os.sep + problem, yhat)

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

    clean_folder('.' + os.sep + 'ms_out_raw')
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
