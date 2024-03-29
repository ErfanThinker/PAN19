# coding=utf-8
from __future__ import division

import glob

from numpy.random import seed
from sklearn.metrics import accuracy_score, f1_score

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

    # Building training set
    train_docs = []
    for candidate in candidates:
        train_docs.extend(read_files(path + os.sep + problem, candidate))

    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    index_2_label_dict = {i: l for i, l in enumerate(set(train_labels))}
    label_2_index_dict = {l: i for i, l in enumerate(set(train_labels))}
    train_labels = [label_2_index_dict[v] for v in train_labels]

    # preparing test set
    ground_truth_file = path + os.sep + problem + os.sep + 'ground-truth.json'
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    test_docs = read_files(path + os.sep + problem, unk_folder, gt)
    test_labels = [label for i, (text, label) in enumerate(test_docs)]

    for ind, v in enumerate(test_labels):
        if v in label_2_index_dict.keys():
            test_labels[ind] = label_2_index_dict[v]
        else:
            test_labels[ind] = len(label_2_index_dict.keys())

    proba = np.load('.' + os.sep + 'ms_out_raw' + os.sep + problem + '.npy')
    yhat = np.argmax(proba, axis=1)
    predictions = [index_2_label_dict[v] for v in yhat]

    # Reject option (used in open-set cases)
    # count = 0
    # for i, p in enumerate(predictions):
    #     sproba = sorted(proba[i], reverse=True)
    #     if sproba[0] - sproba[1] < pt:
    #         predictions[i] = u'<UNK>'
    #         count = count + 1
    # print('\t', count, 'texts left unattributed')

    acc = accuracy_score(test_labels, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
    f_measure = f1_score(test_labels, yhat, average='macro')

    print('my_model_neu Test f-measure: %.3f' % f_measure)


def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()

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
