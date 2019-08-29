# coding=utf-8
# author - Richard Liao
# Dec 26 2016
import logging
from pyclbr import Class

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Lambda
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import activations
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables
from collections import defaultdict

import numpy as np
import os


def gold_softmax(matrix, Gold_Lambda):
    matrix = K.cast(matrix, K.floatx())
    matrix = K.exp(Gold_Lambda * matrix)
    matrix_sum = K.expand_dims(K.sum(matrix, axis=-1), axis=-1)
    matrix_result = K.cast(matrix / (matrix_sum + K.epsilon()), K.floatx())
    return matrix_result

def out_shape(input_shape):
    return input_shape[0] * input_shape[1] * input_shape[2]

def word_cross_entropy(y_true, y_pred):
    # return K.categorical_crossentropy(y_true, y_pred) + K.categorical_crossentropy(y_true, l_att) + K.categorical_crossentropy(y_true, l_att_sent)

    return K.categorical_crossentropy(y_pred, y_true, from_logits=True)

Gold_Lambda = 1
matrix = np.array([[[1,2,1],
              [1,1,4],
              [1,5,1]],
             [[2,1,2],
              [4,2,9],
              [2,6,2]]],
             )
star = np.array([[[1,1,2],
              [1,1,1],
              [1,1,1]],
             [[1,1,2],
              [1,1,1],
              [1,1,1]]],
             )
c = matrix * star
input = Input(((3, 3,)), dtype="float32")
t = gold_softmax(input, 1)

matrix = K.cast(matrix, K.floatx())
star = K.cast(star, K.floatx())

l = K.categorical_crossentropy(star, matrix, from_logits=True)

output = Lambda(gold_softmax, output_shape=out_shape, arguments={"Gold_Lambda": 1})

model = Model(input, input)
model.compile(loss=word_cross_entropy,
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(matrix, star, epochs=1)

pass
