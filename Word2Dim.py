from __future__ import print_function

import multiprocessing as mp
from collections import defaultdict, OrderedDict
from enum import Enum
import numpy as np
from time import sleep
from sklearn import preprocessing
from MyUtils import process_doc
import gc


class WordEmbeddingMode(Enum):
    Normal = 0,
    Scaled = 1,
    # NormalizedVectorized = 2


class Word2Dim(object):

    def __init__(self, lang='en', num_words=None, ignore_index=0, dims=-1):
        self.dims = dims
        self.word_embedding = None
        self.word_index = dict()
        self.num_words = num_words
        self.lang = lang
        self.ignore_index = ignore_index

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

    def __get_word_index(self, word_tuple):
        return self.word_index[word_tuple] if word_tuple in self.word_index.keys() else self.ignore_index

    def fit_transform_texts(self, train_texts, train_labels, tf=None):
        train_texts_plus = [(text, self.lang, i) for i, text in enumerate(train_texts)]
        train_word_set = defaultdict(int)
        print("doc count to process: ", str(len(train_texts_plus)))

        # pool_size = int(mp.cpu_count() / 2) - 2
        pool_size = min(3, int(mp.cpu_count() / 2) - 2)
        pool = mp.Pool(pool_size)
        train_texts_plus = pool.map(process_doc, train_texts_plus)
        pool.close()
        # pool.join()

        # mp.set_start_method('spawn')
        # pool = mp.Pool(pool_size,  maxtasksperchild=1)
        # train_texts_plus = pool.imap(process_doc, train_texts_plus, chunksize=int(pool_size / 2))
        # pool.close()

        # temp = train_texts_plus[:]
        # train_texts_plus = []

        # pool = mp.Pool(pool_size, maxtasksperchild=1)
        # it = pool.imap(process_doc, temp, chunksize=int(pool_size / 2))
        # for res in it:
        #     train_texts_plus.append(res)
        # pool.close()
        # temp = []

        # for doc in temp:
        #     train_texts_plus.append(process_doc(doc))
        #
        # for list_slice_ind in range(0, len(temp), pool_size):
        #     pool = mp.Pool(pool_size, maxtasksperchild=1)
        #     train_texts_plus.extend(pool.map(process_doc, temp[list_slice_ind:list_slice_ind + pool_size]))
        #     pool.close()
        #     pool.join()
        #     gc.collect()
        # assert len(train_texts_plus) == len(temp)

        print('process_doc, done!')
        for train_text in train_texts_plus:
            for tup in train_text:
                train_word_set[tup] += 1

        if tf:
            train_word_set = set([v for v, c in train_word_set.items() if v >= tf])
        else:
            train_word_set = set(list(train_word_set.keys()))
        print('word_set, ready!')
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
                if word_tuple in self.word_index:
                    word_embedding[self.word_index[word_tuple], index] += 1
        # Removing the words used only once
        # word_embedding[word_embedding == 1] = 0

        # calculating embedding for reject option (<UNK>)
        # for vector in word_embedding:
        #     if np.mean(vector[:-1]) > 2:
        #         vector[-1] = np.sum(vector)

        self.word_embedding = word_embedding

        tokenized_and_indexed = []
        for train_text_plus in train_texts_plus:
            tokenized_and_indexed.append([self.__get_word_index(word_pos_tuple) for i, word_pos_tuple in
                                          enumerate(train_text_plus)])
        print('fit_transform_texts is done!')
        return train_texts_plus, tokenized_and_indexed

    def transform(self, texts):
        print("doc count to process: ", str(len(texts)))
        texts_plus = [(text, self.lang, i) for i, text in enumerate(texts)]

        # pool_size = int(mp.cpu_count() / 2) - 2
        pool_size = min(3, int(mp.cpu_count() / 2) - 2)
        pool = mp.Pool(pool_size)
        texts_plus = pool.map(process_doc, texts_plus)
        pool.close()
        # pool.join()

        # pool_size = int(mp.cpu_count() / 2) - 2

        # temp = texts_plus[:]
        # texts_plus = []

        # mp.set_start_method('spawn')
        # pool = mp.Pool(pool_size, maxtasksperchild=1)
        # it = pool.imap(process_doc, temp, chunksize=int(pool_size / 2))
        # for res in it:
        #     texts_plus.append(res)
        # pool.close()
        # temp = []

        # for list_slice_ind in range(0, len(temp), pool_size):
        #     pool = mp.Pool(pool_size, maxtasksperchild=1)
        #     texts_plus.extend(pool.map(process_doc, temp[list_slice_ind:list_slice_ind + pool_size]))
        #     pool.close()
        #     pool.join()
        #     gc.collect()
        # assert len(texts_plus) == len(temp)
        #
        #
        # for doc in temp:
        #     texts_plus.append(process_doc(doc))

        tokenized_and_indexed = []

        for text_plus in texts_plus:
            tokenized_and_indexed.append([self.__get_word_index(word_pos_tuple) for i, word_pos_tuple in
                                          enumerate(text_plus)])
        return texts_plus, tokenized_and_indexed

    def get_word_embedding(self, mode=WordEmbeddingMode.Normal):
        if mode == WordEmbeddingMode.Normal:
            return self.word_embedding

        elif mode == WordEmbeddingMode.Scaled:
            max_abs_scaler = preprocessing.MaxAbsScaler()
            return max_abs_scaler.fit_transform(self.word_embedding)

        else:
            raise ValueError('Provided mode value is not supported.')

    def get_texts_vectorized_and_normalized(self, tokenized_and_indexed_texts):
        data = np.zeros((len(tokenized_and_indexed_texts), self.word_embedding.shape[0],), dtype='float32')
        # print(self.word_index.keys())
        for text_ind, tokenized_and_indexed_text in enumerate(tokenized_and_indexed_texts):
            text_length = len(tokenized_and_indexed_text)
            counts = defaultdict(float)
            for i in tokenized_and_indexed_text:
                counts[i] += 1
            for word_index, word_count in counts.items():
                data[text_ind, word_index] = word_count / text_length

        return data
