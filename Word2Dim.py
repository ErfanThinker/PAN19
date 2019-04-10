from __future__ import print_function

import argparse
import codecs
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict
import numpy as np
from sklearn import preprocessing

from MyUtils import read_files, process_doc


def __tuple_list_2_dict(da_list):
    out = defaultdict(list)
    for val in da_list:
        out[val[1]].append(val[0])
    return out


def __create_author_2_word_pos_dict(authors, texts):
    out = defaultdict(list)
    for idx, aut in enumerate(authors):
        out[aut].extend(texts[idx])
    return out


def generate_scores(problem_dir, lang):
    infoproblem = problem_dir + os.sep + 'problem-info.json'
    candidates = []
    pool = mp.Pool(int(mp.cpu_count() / 2) - 1)
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    train_docs = []
    for candidate in candidates:
        train_docs.extend(read_files(problem_dir, candidate))
    train_texts = [(text, lang, i) for i, (text, label) in enumerate(train_docs)]
    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    train_word_set = set()
    print("doc count to process: ", str(len(train_texts)))

    train_texts = pool.map(process_doc, train_texts)
    print('process_doc, done!')
    for train_text in train_texts:
        train_word_set.update(train_text)
    train_word_set = list(train_word_set)
    train_word_set.insert(0, ('<PAD>', 'NAP'))
    # print(train_texts)
    print(str(len(train_word_set)))
    # This will give something like this: {label_0:[text_0, text_1,...], label_1: [...] , ... }
    author_dict = __create_author_2_word_pos_dict(train_labels, train_texts)
    # ppr = pp.PrettyPrinter(indent=4)
    # ppr.pprint(author_dict)
    print(author_dict)

    # word_embedding = np.zeros((len(train_word_set), len(author_dict.keys()) + 1,))
    word_embedding = np.zeros((len(train_word_set), len(author_dict.keys()),))
    for i, (label, word_tuples) in enumerate(author_dict.items()):
        for word_tuple in word_tuples:
            word_embedding[train_word_set.index(word_tuple), i] += 1
    # Removing the words used only once
    word_embedding[word_embedding == 1] = 0

    # calculating embedding for reject option (<UNK>)
    # for vector in word_embedding:
    #     if np.mean(vector[:-1]) > 2:
    #         vector[-1] = np.sum(vector)

    max_abs_scaler = preprocessing.MaxAbsScaler()
    word_embedding = max_abs_scaler.fit_transform(word_embedding)
    return word_embedding, train_word_set
