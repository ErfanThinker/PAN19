# coding=utf-8

# import logging
# from keras import models
# from keras import layers
# from keras import Input
from Word2Dim import Word2Dim
from collections import defaultdict

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'wtf?'
]
labels = [1, 2, 2, 1, 3]


# w2d = Word2Dim()
# train_texts_plus, tokenized_and_indexed = w2d.fit_transform_texts(corpus, [1, 1, 2, 2], 'en')
# print(w2d.get_texts_vectorized_and_normalized(tokenized_and_indexed))

def shuffle_docs2(texts, labels):
    new_texts = []
    new_labels = []
    validation_index = 0
    records_dict = defaultdict(list)
    training_ratio = .7
    for i, l in enumerate(labels):
        records_dict[l].append(texts[i])
    print(records_dict)
    keep_sizes = []
    for label, texts in records_dict.items():
        keep_sizes.append(round(len(texts) * training_ratio))
        new_texts.extend(texts[:keep_sizes[-1]])
        new_labels.extend([label] * keep_sizes[-1])

    validation_index = len(new_labels)

    for i, (label, texts) in enumerate(records_dict.items()):
        bases = texts[keep_sizes[i]:]
        # add original records
        new_texts.extend(bases)
        new_labels.extend([label] * len(bases))

        # generate new docs based on unused training docs as bases and seen training docs
        additionals = texts[:keep_sizes[i]]
        for base in bases:
            tokenized_text = base.split(' ')
            first_half = tokenized_text[:int(len(tokenized_text) / 2)]
            second_half = tokenized_text[int(len(tokenized_text) / 2):]
            for additional in additionals:
                tokenized_other_text = additional.split(' ')
                first_half_of_other_text = tokenized_other_text[:int(len(tokenized_other_text) / 2)]
                second_half_of_other_text = tokenized_other_text[int(len(tokenized_other_text) / 2):]
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(second_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(second_half_of_other_text))
                new_labels.extend([label] * 4)

    return new_texts, new_labels, validation_index


print(shuffle_docs2(corpus, labels))
# for i in range(0, len(corpus), 4):
#     print(corpus[i:i + 4])
#
# l = list(range(7))
# print(round(4.3))
