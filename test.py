# coding=utf-8
# import logging
# from keras import models
# from keras import layers
# from keras import Input
from collections import defaultdict


def tuple_list_2_dict(da_list):
    out = defaultdict(list)
    for val in da_list:
        out[val[1]].append(val[0])
    return out


my_list = [(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b')]
print(tuple_list_2_dict(my_list))
