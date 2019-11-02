# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

# os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from nltk import tokenize

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

#
label_list =["First Party Collection", "Data Retention", "Cookies and Similar Technologies", "First Party Use", "Links",
             "User Control", "Introductory/Generic", "Data Security", "Third Party Sharing and Collection", "User Right",
             "Internal and Specific Audiences", "Data Transfer", "Policy Change", "Legal Basis", "Privacy Contact Information"]


for label in label_list:
# label = "15 classification"
    model_name = "RNN"
    classification_name = label.replace(" ", "_").replace("/", "_")
    GLOVE_DIR = "./word_embedding/"
    embedding_file = 'fasttext_embedding.vec'
    train_path = './dataset/' + classification_name + '_train_data.tsv'
    dev_path = './dataset/' + classification_name + '_dev_data.tsv'
    test_path = './dataset/' + classification_name + '_test_data.tsv'

    save_model_path = "./model/" + model_name + classification_name + "_model.h5"
    log_path = "./log/" + model_name + classification_name + ".log"
    import logging
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - '+ classification_name +' -%(levelname)s: %(message)s'
                        )

    MAX_SEQUENCE_LENGTH = 800
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    MAX_NB_WORDS = 60000
    Label_Size = 2

    epochs = 30
    batch_size = 512



    data_train = pd.read_csv(train_path, sep='\t')
    data_dev = pd.read_csv(dev_path, sep='\t')
    data_test = pd.read_csv(test_path, sep='\t')

    # data_train = pd.read_csv('~/Testground/data/imdb/labeledTrainData.tsv', sep='\t')
    train_split_num = len(data_train)
    dev_split_num = len(data_dev) + train_split_num

    data_train = pd.DataFrame(pd.concat([data_train, data_dev, data_test], ignore_index=True))

    print data_train.shape

    texts = []
    labels = []

    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        texts.append(clean_str(text.get_text().encode('ascii','ignore')))
        labels.append(data_train.sentiment[idx])


    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:train_split_num]
    y_train = labels[:train_split_num]

    x_val = data[train_split_num:dev_split_num]
    y_val = labels[train_split_num:dev_split_num]

    x_test = data[dev_split_num:]
    y_test = labels[dev_split_num:]

    print('Traing and validation set number of positive and negative reviews')
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
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(2, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - Bidirectional LSTM - " + classification_name)
    logging.info("model fitting - Bidirectional LSTM - " + classification_name)
    logging.info(model.summary())

    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch = epochs, batch_size=batch_size)

    logging.info(label + "\n")
    hist_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    logging.info(hist.history)
    logging.info(hist.epoch)
    logging.info(str(hist_test))
    logging.info("Evaluate Recall Precision F1:\n")
    hist_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    model.save(save_model_path)

