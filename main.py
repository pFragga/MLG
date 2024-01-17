#!/usr/bin/env python

# import matplotlib as plt
# import pandas as pd
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in
                        x_train])
    x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in
                       x_test])
