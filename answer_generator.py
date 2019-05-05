# coding=utf-8
from __future__ import division

import glob

from numpy.random import seed


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

    proba = np.load('.' + os.sep + 'ms_out_raw' + os.sep + problem + '.npy')
    yhat = np.argmax(proba, axis=1)
    predictions = [index_2_label_dict[v] for v in yhat]

    # Reject option (used in open-set cases)
    count = 0
    for i, p in enumerate(predictions):
        sproba = sorted(proba[i], reverse=True)
        if sproba[0] - sproba[1] < pt:
            predictions[i] = u'<UNK>'
            count = count + 1
    print('\t', count, 'texts left unattributed')
    # Saving output data
    out_data = []
    unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
    pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
    for i, v in enumerate(predictions):
        out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
    with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
        json.dump(out_data, f, indent=4)
    print('\t', 'answers saved to file', 'answers-' + problem + '.json')


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
    parser.add_argument('-pt', type=float, default=0.6, help='probability threshold for the reject option (default=0.1')
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
