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
    'wtf?'
]
labels = [1, 2, 2, 1, 3]
# w2d = Word2Dim()
# train_texts_plus, tokenized_and_indexed = w2d.fit_transform_texts(corpus, [1, 1, 2, 2], 'en')
# print(w2d.get_texts_vectorized_and_normalized(tokenized_and_indexed))

def shuffle_docs(texts, labels):
    new_texts = []
    new_labels = []
    for ind, text in enumerate(texts):
        if ind != len(texts) - 1:
            tokenized_text = text.split(' ')
            first_half = tokenized_text[:int(len(tokenized_text) / 2)]
            second_half = tokenized_text[int(len(tokenized_text) / 2):]
            remaining_text_from_this_writer = [text for index, text in enumerate(texts)
                                               if index > ind and labels[index] == labels[ind]]
            for other_text in remaining_text_from_this_writer:
                tokenized_other_text = other_text.split(' ')
                first_half_of_other_text = tokenized_other_text[:int(len(tokenized_other_text) / 2)]
                second_half_of_other_text = tokenized_other_text[int(len(tokenized_other_text) / 2):]
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(second_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(second_half_of_other_text))
                new_labels.extend([labels[ind]] * 4)

    new_texts.extend(texts)
    new_labels.extend(labels)
    return new_texts, new_labels

# print(shuffle_docs(corpus, labels))
for i in range(0,len(corpus),4):
    print(corpus[i:i+4])