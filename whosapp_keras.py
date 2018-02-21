#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mathias.mueller@uzh.ch
from __future__ import unicode_literals

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict
import codecs
import sys
import re
import logging

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


def preprocess(path=None,
               exclude_authors=[],
               rename_authors=[],
               samples_threshold=None):
    """
    Reads lines from the raw Whatsapp data dump and converts
    them into training examples. Date and time are currently extracted
    but ignored.
    """
    X, y = [], []

    d = defaultdict(list)
    previous = None, None, None

    if path is not None:
        data = codecs.open(path, "r", "UTF-8")
    else:
        data = sys.stdin

    for line in data:
        # skip empty lines
        if line.strip() == "":
            continue

        # skip media notifications
        if "<Media omitted>" in line:
            continue

        if re.findall("\d{2}/\d{2}/\d{4}, \d{2}:\d{2}", line):
            # line with timestamp and author

            parts = line.strip().split("-")
            date, time = [p.strip() for p in parts[0].split(",")]

            if ":" in parts[1]:
                # line with actual text
                author_content = parts[1].split(":")
                author = author_content[0].strip()
                content = u" ".join(
                    [x.strip() for x in author_content[1:] if x.strip() != ""])
            else:
                # line with Whatsapp notification
                continue  # for now
        else:
            # line without timestamp, continuation with same author
            date, time, author = previous
            content = line.strip()

        if author in exclude_authors:
            continue
        else:
            if author in rename_authors:
                author = rename_authors[author]

            X.append(content)
            y.append(author)

            d[author].append((date, time, content))

        previous = date, time, author

    if samples_threshold:
        deletes = []
        for k, v in d.iteritems():
            if len(v) < samples_threshold:
                deletes.append(k)
        for k in deletes:
            del d[k]

    logging.debug("Messages with actual content by author:")
    for k, v in d.iteritems():
        logging.debug("%s %d" % (k, len(v)))
    logging.debug("Total messages: %d\n" %
                  sum([len(v) for v in d.values()]))

    return X, y


def build_model():
    X, y = preprocess()

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    categorical_y = to_categorical(encoded_y)

    X_train, X_test, y_train, y_test = train_test_split(X, categorical_y, test_size=0.1, random_state=None)

    # length in characters
    max_seq_len = 100

    # vocabulary size
    num_words = 10000

    X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

    # create the model

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(num_words, embedding_vecor_length,
                        input_length=max_seq_len))

    num_classes = len(encoder.classes_)

    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=3, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


level = logging.DEBUG
logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

build_model()
