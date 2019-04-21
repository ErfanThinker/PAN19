# coding=utf-8

# import logging
# from keras import models
# from keras import layers
# from keras import Input
from Word2Dim import Word2Dim

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
w2d = Word2Dim()
train_texts_plus, tokenized_and_indexed = w2d.fit_transform_texts(corpus, [1, 1, 2, 2], 'en')
print(w2d.get_texts_vectorized_and_normalized(tokenized_and_indexed))
