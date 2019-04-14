# coding=utf-8
import tensorflow as tf
# import logging
# from keras import models
# from keras import layers
# from keras import Input
from collections import defaultdict
import numpy as np
from sklearn import preprocessing


def tuple_list_2_dict(da_list):
    out = defaultdict(list)
    for val in da_list:
        out[val[1]].append(val[0])
    return out


my_list = [(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b')]
print(tuple_list_2_dict(my_list))

a = np.array([[1, 2, 3],
              [1, 5, 6],
              [1, 7, 9]])
scaled_train_data = preprocessing.maxabs_scale(a[:-1, ], axis=0)
print(scaled_train_data)
print(scaled_train_data.mean(axis=0))
print(a.mean(axis=0))
print(a.std(axis=0))

print(tf.VERSION)

t = {'a': 1, 'b': 5}
print(max(t.values()))