# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os

import numpy as np
from gensim.models import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from sklearn.cross_validation import train_test_split


DATA_DIR = "../data/comp_data"
MODEL_DIR = "../data/models"
WORD2VEC_BIN = "../data/GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_EMBED_SIZE = 300

QA_TRAIN_FILE = "MiddleSchool-DMC-Train.csv"

QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 20

## extract data

print("Loading and formatting data...")
qapairs = kaggle.get_question_answer_pairs(os.path.join(DATA_DIR, QA_TRAIN_FILE))
question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
seq_maxlen = max([question_maxlen, answer_maxlen])

word2idx = kaggle.build_vocab([], qapairs, [])
vocab_size = len(word2idx) + 1  # include mask character 0

Xq, Xa, Y = kaggle.vectorize_qapairs(qapairs, word2idx, seq_maxlen)
Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xqtrain.shape, Xqtest.shape, Xatrain.shape, Xatest.shape,
      Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
print("Loading Word2Vec model and generating embedding matrix...")
word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_BIN, binary=True)
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)

print("Building model...")
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   weights=[embedding_weights], mask_zero=True))
print(QA_EMBED_SIZE, seq_maxlen)
qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, input_shape=(seq_maxlen, WORD2VEC_EMBED_SIZE),
                            return_sequences=False), merge_mode="sum"))
qenc.add(Dropout(0.3))

aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   weights=[embedding_weights], mask_zero=True))
aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, input_shape=(seq_maxlen, WORD2VEC_EMBED_SIZE),
                            return_sequences=False), merge_mode="sum"))
aenc.add(Dropout(0.3))

model = Sequential()
model.add(Merge([qenc, aenc], mode="sum"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Training...")
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "qa-blstm-best.hdf5"),
    verbose=1, save_best_only=True)
model.fit([Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation...")
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))

model.save_weights(os.path.join(MODEL_DIR, "qa-blstm-final.hdf5"))
with open(os.path.join(MODEL_DIR, "qa-blstm-model.json"), "w") as fjson:
    fjson.write(model.to_json())

model.load_weights(filepath=os.path.join(MODEL_DIR, "qa-blstm-best.hdf5"))
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
