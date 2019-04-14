from __future__ import print_function

import multiprocessing as mp
from collections import defaultdict, OrderedDict

import numpy as np
from sklearn import preprocessing

from MyUtils import process_doc


class Word2Dim(object):

    def __init__(self, num_words=None, dims=-1):
        self.dims = dims
        self.word_embedding = None
        self.word_index = dict()
        self.num_words = num_words
        self.lang = ''

    def __tuple_list_2_dict(self, da_list):
        out = defaultdict(list)
        for val in da_list:
            out[val[1]].append(val[0])
        return out

    def __create_author_2_word_pos_dict(self, authors, texts):
        out = defaultdict(list)
        for idx, aut in enumerate(authors):
            out[aut].extend(texts[idx])

        # sort on author names (keys) before returning
        return OrderedDict(sorted(out.items(), key=lambda t: t[0]))

    def fit_transform_texts(self, train_texts, train_labels, lang):
        self.lang = lang
        train_texts_plus = [(text, lang, i) for i, text in enumerate(train_texts)]
        train_word_set = set()
        print("doc count to process: ", str(len(train_texts_plus)))

        pool = mp.Pool(int(mp.cpu_count() / 2) - 1)
        train_texts_plus = pool.map(process_doc, train_texts_plus)
        print('process_doc, done!')
        for train_text in train_texts_plus:
            train_word_set.update(train_text)

        self.word_index = {v: i + 1 for i, v in enumerate(train_word_set)}
        # train_word_set.insert(0, ('<pad>', 'NAP'))
        # print(str(len(train_word_set)))
        # This will give something like this: {label_0:[text_0, text_1,...], label_1: [...] , ... }
        author_dict = self.__create_author_2_word_pos_dict(train_labels, train_texts_plus)

        # word_embedding = np.zeros((len(train_word_set), len(author_dict.keys()) + 1,))
        word_embedding = np.zeros((len(train_word_set) + 1, len(author_dict.keys()),), dtype='float32')

        # author_dict is sorted on author names so the index will work out
        for index, (label, word_tuples) in enumerate(author_dict.items()):
            for word_tuple in word_tuples:
                word_embedding[self.word_index[word_tuple], index] += 1
        # Removing the words used only once
        # word_embedding[word_embedding == 1] = 0

        # calculating embedding for reject option (<UNK>)
        # for vector in word_embedding:
        #     if np.mean(vector[:-1]) > 2:
        #         vector[-1] = np.sum(vector)

        max_abs_scaler = preprocessing.MaxAbsScaler()
        word_embedding = max_abs_scaler.fit_transform(word_embedding)
        self.word_embedding = word_embedding
        tokenized_and_indexed = []
        for train_text_plus in train_texts_plus:
            tokenized_and_indexed.append([self.word_index[word_pos_tuple] for i, word_pos_tuple in
                                          enumerate(train_text_plus)])
        return train_texts_plus, tokenized_and_indexed

    def transform(self, texts):
        texts_plus = [(text, self.lang, i) for i, text in enumerate(texts)]
        pool = mp.Pool(int(mp.cpu_count() / 2) - 1)
        texts_plus = pool.map(process_doc, texts_plus)
        tokenized_and_indexed = []
        ignore_index = len(self.word_index) + 1
        for text_plus in texts_plus:
            tokenized_and_indexed.append([self.word_index[word_pos_tuple] if word_pos_tuple in self.word_index.keys()
                                          else ignore_index for i, word_pos_tuple in
                                          enumerate(text_plus)])
        return texts_plus, tokenized_and_indexed