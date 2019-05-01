from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences


def pad_seqs(seq, maxlen=1):
    return pad_sequences(seq, maxlen=maxlen)


def encode_labels(labels):
    return to_categorical(labels)
