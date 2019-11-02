# coding=utf-8
# author - Richard Liao
# Dec 26 2016
import h5py
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
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

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

result = []
model_name= "HATT_"
eval_path = "./eval/"+ model_name  + ".csv"
label_list =["First Party Collection", "Data Retention", "Cookies and Similar Technologies", "First Party Use", "Links",
             "User Control", "Introductory/Generic", "Data Security", "Third Party Sharing and Collection", "User Right",
             "Internal and Specific Audiences", "Data Transfer", "Policy Change", "Legal Basis", "Privacy Contact Information"]

for label in label_list:
# label = "15 classification"

    classification_name = label.replace(" ", "_").replace("/", "_")
    
    GLOVE_DIR = "./word_embedding/"
    embedding_file = 'fasttext_embedding.vec'
    train_path = './data/' + classification_name + '_train_data.tsv'
    dev_path = './data/' + classification_name + '_dev_data.tsv'
    test_path = './data/' + classification_name + '_test_data.tsv'
    # eval_path = "./data_status/"+ model_name + classification_name + ".csv"
    
    save_model_path = "./model/" + model_name + classification_name + "_model.h5"
    log_path = "./log/" + model_name + classification_name + ".log"
    
    MAX_SENT_LENGTH = 150
    MAX_SENTS = 10
    MAX_NB_WORDS = 60000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    Label_Size = 2
    
    epochs = 30
    batch_size = 512
    import logging
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - '+ classification_name +' -%(levelname)s: %(message)s'
                        )
    
    
    # read data from tsv
    data_train = pd.read_csv(train_path, sep='\t')
    data_dev = pd.read_csv(dev_path, sep='\t')
    data_test = pd.read_csv(test_path, sep='\t')
    
    train_split_num = len(data_train)
    dev_split_num = len(data_dev) + train_split_num
    
    data_train = pd.DataFrame(pd.concat([data_train, data_dev, data_test], ignore_index=True))
    print data_train.shape
    
    from nltk import tokenize
    
    reviews = []
    labels = []
    texts = []
    
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
    
        labels.append(data_train.sentiment[idx])
    
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    
    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    
    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
    
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    true_label = np.asarray(labels[dev_split_num:])
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    # load data for model to train
    
    x_train = data[:train_split_num]
    y_train = labels[:train_split_num]
    
    x_val = data[train_split_num:dev_split_num]
    y_val = labels[train_split_num:dev_split_num]
    
    x_test = data[dev_split_num:]
    y_test = labels[dev_split_num:]
    
    
    
    
    print('Number of positive and negative reviews in traing and validation set')
    print y_train.sum(axis=0)
    print y_val.sum(axis=0)
    
    GLOVE_DIR = "./word_embedding"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'fasttext_embedding.vec'))
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
    
    
    # building Hierachical Attention network
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
    
    
    class AttLayer(Layer):
        def __init__(self, attention_dim):
            self.init = initializers.get('normal')
            self.supports_masking = True
            self.attention_dim = attention_dim
            super(AttLayer, self).__init__()
    
        def build(self, input_shape):
            assert len(input_shape) == 3
            self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
            self.b = K.variable(self.init((self.attention_dim, )))
            self.u = K.variable(self.init((self.attention_dim, 1)))
            self.trainable_weights = [self.W, self.b, self.u]
            super(AttLayer, self).build(input_shape)
    
        def compute_mask(self, inputs, mask=None):
            print("mask", mask)
            return mask
    
        def call(self, x, mask=None):
            # size of x :[batch_size, sel_len, attention_dim]
            # size of u :[batch_size, attention_dim]
            # uit = tanh(xW+b)
            uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
            ait = K.dot(uit, self.u)
            ait = K.squeeze(ait, -1)
    
            ait = K.exp(ait)
    
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                ait *= K.cast(mask, K.floatx())
            ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            ait = K.expand_dims(ait)
            weighted_input = x * ait
            output = K.sum(weighted_input, axis=1)
    
            return output
    
        def compute_output_shape(self, input_shape):
            print(input_shape)
            return (input_shape[0], input_shape[-1])
    
    
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)
    print(type(l_att))
    sentEncoder = Model(sentence_input, l_att)
    
    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(Label_Size, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)
    logging.info(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    logging.info("model fitting - Hierachical attention network - " + classification_name)
    print("model fitting - Hierachical attention network - " + classification_name)
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size)
    logging.info(label + "\n")
    hist_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    logging.info(hist.history)
    logging.info(hist.epoch)
    logging.info(str(hist_test))
    logging.info("Evaluate Recall Precision F1:\n")
    
    # 中间层输出
    # ait_layer = Model(inputs=model.input, outputs=model.get_layer(''))
    
    hist_pre = model.predict(x_test, batch_size = batch_size)
    predic_classes = np.argmax(hist_pre, axis=1)
    from sklearn.metrics import f1_score, precision_score, recall_score
    a = [label, precision_score(true_label, predic_classes), recall_score(true_label, predic_classes), f1_score(true_label, predic_classes)]
    logging.info(str(a))
    result.append(a)
    # result = [[precision_score(true_label, predic_classes), recall_score(true_label, predic_classes), f1_score(true_label, predic_classes)]]
    # logging.info()
    # result = model.predict(x_train)
    
    # ---------------------------------------------------------------------------
    # --------------------- For save Model --------------------------------------
    # ---------------------------------------------------------------------------
    
    
    # save the weight maybe it can success
    # 1
    # file = h5py.File(save_model_path+"1.h5", 'w')
    # weight = []
    # for i in range(len(file.keys())):
    #     weight.append(file['weight' + str(i)][:])
    # model.set_weights(weight)
    #2
    # json_string = model.to_json()
    model.save_weights(save_model_path+"weights.h5")
    #3
    model.save(save_model_path+"model.h5")
    
    
    # ---------------------------------------------------------------------------
    # --------------------- For load Model --------------------------------------
    # ---------------------------------------------------------------------------
    # Load weight
    # 1
    # file = h5py.File(save_model_path+"1.h5", "r")
    # weight = model.get_weights()
    # for i in range(len(weight)):
    #     file.create_dataset('weight' + str(i), data=weight[i])
    # file.close()
    # # 2
    #
    # from keras.models import model_from_json
    #
    # # model = model_from_json(json_string)
    # model.load_weights(save_model_path+"weights.h5")
    # #
    # from keras.models import load_model
    # model = load_model(save_model_path+"model.h5")
    # pd.DataFrame(result, columns=["Label", "P", "R", "F"]).to_csv(eval_path, header="True")
pd.DataFrame(result, columns=["P", "R", "F"]).to_csv(eval_path, header="True")
