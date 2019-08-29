# coding=utf-8
# author - Richard Liao
# Dec 26 2016
import logging
import tensorflow
from pyclbr import Class

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
from nltk.stem import PorterStemmer
from nltk import tokenize
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
label_list =["First Party Collection", "Data Retention", "Cookies and Similar Technologies", "First Party Use", "Links",
             "User Control", "Introductory/Generic", "Data Security", "Third Party Sharing and Collection", "User Right",
             "Internal and Specific Audiences", "Data Transfer", "Policy Change", "Legal Basis", "Privacy Contact Information"]

for label in label_list:

    model_name = "LBSA_"
    classification_name = label.replace(" ", "_").replace("/", "_")

    GLOVE_DIR = "./word_embedding/"
    embedding_file = 'fasttext_embedding.vec'
    train_path = './dataset/' + classification_name + '_train_data.tsv'
    dev_path = './dataset/' + classification_name + '_dev_data.tsv'
    test_path = './dataset/' + classification_name + '_test_data.tsv'

    save_model_path = "./model/" + model_name + classification_name + "_model.h5"
    log_path = "./log/" + model_name + ".log"

    MAX_SENT_LENGTH = 150
    MAX_SENTS = 10
    MAX_NB_WORDS = 800
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    Label_Size = 2

    att_dropout_in = 0.5
    # em_droup_in = 0.5
    lstm_drop = 0.5
    Gold_word_lambda = 1
    Gold_sent_lambda = 1
    # loss_word_balance = 0.001
    # loss_sent_balance = 0.05
    loss_word_balance = 0.001
    loss_sent_balance = 0.05

    word_lstm_hidden_size = 100
    sent_lstm_hidden_size = 100
    att_size = 100

    epochs = 30
    batch_size = 512

    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - ' + classification_name +' -%(levelname)s: %(message)s'
                        )


    def word_cross_entropy(y_true, y_pred):
        # return K.categorical_crossentropy(y_true, y_pred) + K.categorical_crossentropy(y_true, l_att) + K.categorical_crossentropy(y_true, l_att_sent)
        return K.sum(K.categorical_crossentropy(y_pred, y_true, from_logits=True), axis=-1)


    def clean_str(string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()


    def get_gold_score(wd, attention_words):
        ps = PorterStemmer()
        return 1 if ps.stem(wd) in attention_words else 1e-8


    def gold_softmax(matrix, Gold_Lambda):
        matrix = matrix.astype(np.float32)
        matrix = np.exp(Gold_Lambda * matrix)
        matrix_sum = np.expand_dims(np.sum(matrix, axis=-1), axis=-1)
        matrix_result = (matrix / (matrix_sum + 1e-8)).astype(np.float32)
        return matrix_result


    def encode_gold_sent(gold_word_sd):
        return np.mean(gold_word_sd.astype(np.float32), axis=-1)

    # data load

    data_train = pd.read_csv(train_path, sep='\t')
    data_dev = pd.read_csv(dev_path, sep='\t')
    data_test = pd.read_csv(test_path, sep='\t')

    train_split_num = len(data_train)
    dev_split_num = len(data_dev) +  train_split_num


    data_train = pd.DataFrame(pd.concat([data_train, data_dev, data_test], ignore_index=True))
    print data_train.shape

    reviews = []
    labels = []
    texts = []

    reviews_dev = []
    # labels_dev = []
    texts_dev = []

    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(data_train.sentiment[idx])

    # for idx in range(data_dev.review.shape[0]):
    #     text = BeautifulSoup(data_dev.review[idx])
    #     text = clean_str(text.get_text().encode('ascii', 'ignore'))
    #     texts_dev.append(text)
    #     sentences = tokenize.sent_tokenize(text)
    #     reviews_dev.append(sentences)
    #     labels_dev.append(data_train.sentiment[idx])

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    # tokenizer_dev = Tokenizer(nb_words=MAX_NB_WORDS)
    # tokenizer_dev.fit_on_texts(texts_dev)
    # word_index_dev = tokenizer_dev.word_index

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    # dev = np.zeros((len(texts_dev), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    # read gold attention keywords
    ps = PorterStemmer()
    attention_words = [(ps.stem(wd.strip()) for wd in open("./dataset/attention_score.txt").readlines())]

    gold_word_sd = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH:
                        gold_word_sd[i, j, k] = get_gold_score(word, attention_words)
                        k = k + 1

    # gold_word_sd_dev = np.zeros((len(texts_dev), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    # for i, sentences in enumerate(reviews_dev):
    #     for j, sent in enumerate(sentences):
    #         if j < MAX_SENTS:
    #             wordTokens = text_to_word_sequence(sent)
    #             k = 0
    #             for _, word in enumerate(wordTokens):
    #                 if k < MAX_SENT_LENGTH:
    #                     gold_word_sd_dev[i, j, k] = get_gold_score(word, attention_words)
    #                     k = k + 1


    gold_word_attention = gold_softmax(gold_word_sd, Gold_word_lambda)
    gold_sent_sd = encode_gold_sent(gold_word_sd)
    gold_sent_attention = gold_softmax(gold_sent_sd, Gold_sent_lambda)

    #
    # gold_word_attention_dev = gold_softmax(gold_word_sd_dev, Gold_word_lambda)
    # gold_sent_sd_dev = encode_gold_sent(gold_word_sd_dev)
    # gold_sent_attention_dev = gold_softmax(gold_sent_sd_dev, Gold_sent_lambda)

    # word 2 idx
    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    # for i, sentences in enumerate(reviews_dev):
    #     for j, sent in enumerate(sentences):
    #         if j < MAX_SENTS:
    #             wordTokens = text_to_word_sequence(sent)
    #             k = 0
    #             for _, word in enumerate(wordTokens):
    #                 if k < MAX_SENT_LENGTH and tokenizer_dev.word_index[word] < MAX_NB_WORDS:
    #                     dev[i, j, k] = tokenizer_dev.word_index[word]
    #                     k = k + 1

    word_index = tokenizer.word_index
    # word_index_dev = tokenizer_dev.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    # labels_dev = to_categorical(np.asarray(labels_dev))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:train_split_num]
    y_train = labels[:train_split_num]
    word_attention_star = gold_word_attention[:train_split_num]
    sent_attention_star = gold_sent_attention[:train_split_num]

    x_val = data[train_split_num:dev_split_num]
    y_val = labels[train_split_num:dev_split_num]
    word_attention_star_val = gold_word_attention[train_split_num:dev_split_num]
    sent_attention_star_val = gold_sent_attention[train_split_num:dev_split_num]

    x_test = data[dev_split_num:]
    y_test = labels[dev_split_num:]
    word_attention_star_test = gold_word_attention[dev_split_num:]
    sent_attention_star_test = gold_sent_attention[dev_split_num:]


    # model

    print('Number of positive and negative reviews in traing and validation set')
    print y_train.sum(axis=0)
    print y_val.sum(axis=0)

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, embedding_file))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True)


    class Attention_Layer(Layer):
        def __init__(self, units,
                     activation=None,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     attention_level=None,
                     **kwargs):
            self.init = initializers.get('normal')
            self.supports_masking = True
            self.attention_level = attention_level
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.units = units
            self.activation = activations.get(activation)
            # self.attention_dim = attention_dim

            super(Attention_Layer, self).__init__(**kwargs)

        def build(self, input_shape):
            assert len(input_shape) == 3

            input_dim = input_shape[-1]

            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel_att')

            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias_att')

            self.u = self.add_weight(shape=(self.units, 1),
                                     initializer='normal',
                                     name='u')

            super(Attention_Layer, self).build(input_shape)

        def compute_mask(self, inputs, mask=None):
            return mask

        def call(self, inputs, mask=None):
            # size of x :[batch_size, sel_len, attention_dim]
            # size of u :[batch_size, attention_dim]
            # uit = tanh(xW+b)
            inputs = K.dropout(inputs, level=att_dropout_in, seed=666)

            uit = K.tanh(K.bias_add(K.dot(inputs, self.kernel), self.bias))
            ait = K.dot(uit, self.u)
            ait = K.squeeze(ait, -1)
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                ait *= K.cast(mask, K.floatx())
            output = self.activation(ait)
            return output

        def compute_output_shape(self, input_shape):
            output_dim = 0
            if self.attention_level == "word":
                output_dim = MAX_SENT_LENGTH
            elif self.attention_level == "sent":
                output_dim = MAX_SENTS
            return (input_shape[0], output_dim)


    class Encoder_Message(Layer):
        def __init__(self, **kwargs):

            self.supports_masking = True

            super(Encoder_Message, self).__init__(**kwargs)

        def build(self, input_shape):
            assert isinstance(input_shape, list)
            super(Encoder_Message, self).build(input_shape)

        def compute_mask(self, inputs, mask=None):
            # print("mask", mask)
            return mask

        # def call(self, inputs, mask=None):
        def call(self, x, mask=None):
            # x = K.dropout(x, level=em_droup_in)
            assert isinstance(x, list)
            attention, lstm_state = x
            ait = K.expand_dims(attention)
            weighted_input = lstm_state * ait
            if len(lstm_state.shape) > 3:
                axis = 2
            else:
                axis = 1
            output = K.sum(weighted_input, axis=axis)
            return output

        def compute_output_shape(self, input_shape):
            assert isinstance(input_shape, list)
            shape_attention, shape_lstm = input_shape
            if len(shape_lstm) > 3:
                return (shape_lstm[0], shape_attention[1], shape_lstm[-1])
            return (shape_lstm[0], shape_lstm[-1])


    def encoder_HATT(AttLayer, LstmLayer):
        ait = K.expand_dims(AttLayer)
        weighted_input = LstmLayer * ait
        output = K.sum(weighted_input, axis=1)
        return output


    def att_sum(x):
        return K.sum(x, axis=1)


    def word_att_output_shape(input_shape):
        return (input_shape[0], input_shape[-1])

    # Embedding
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    # word to sentence
    l_lstm = Bidirectional(LSTM(word_lstm_hidden_size, return_sequences=True, dropout=lstm_drop))(embedded_sequences)

    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    word_att_all = TimeDistributed(Attention_Layer(units=att_size, attention_level='word', name="att_word", activation='softmax'))(review_encoder)


    encode_sent_message = Encoder_Message()([word_att_all, review_encoder])
    # sentence to doc
    l_lstm_sent = Bidirectional(LSTM(sent_lstm_hidden_size, return_sequences=True, dropout=lstm_drop))(encode_sent_message )

    att_sent_layer = Attention_Layer(units=att_size, attention_level='sent', name="att_sent", activation='softmax')
    att_sent = att_sent_layer(l_lstm_sent)
    w = att_sent_layer.get_weights()

    l_att_sent = Encoder_Message()([att_sent, l_lstm_sent])
    preds = Dense(units=Label_Size, activation='softmax', name='preds')(l_att_sent)  # (?, 2)

    model = Model(review_input, outputs=[preds, att_sent, word_att_all])

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[1, loss_sent_balance, loss_word_balance],
                  optimizer='adam',
                  metrics=['acc'])

    logging.info(classification_name+ "model fitting - LBSA\n")
    print("model fitting - LBSA")

    hist = model.fit(x_train, [y_train, sent_attention_star, word_attention_star],
                     epochs=epochs, batch_size=batch_size,
                     validation_data=(x_val, [y_val, sent_attention_star_val, word_attention_star_val]))

    hist_test = model.evaluate(x_test, [y_test, sent_attention_star_test, word_attention_star_test], batch_size=batch_size)
    logging.info(label + "\n")
    logging.info(hist.history)
    logging.info(hist.epoch)
    logging.info(str(hist_test))
    logging.info("\n")
    # result = model.predict(x_train)

