#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mathias.mueller@uzh.ch

from __future__ import unicode_literals

from pandas import DataFrame
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier


from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics

from util import remove_emojis

import logging
import argparse
import numpy
import re
import json
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf8')

numpy.random.seed(6)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


class Trainer(object):
    """
    Reads raw Whatsapp chat data and trains a classifier.
    """

    def __init__(self, model, data, vectorizer, vectorizer_ngram_order,
                 vectorizer_analyzer, remove_emojis, samples_threshold, exclude_authors,
                 rename_authors, class_weight, classifier, evaluation, cv_folds,
                 test_fold_size, f1_averaging):
        """
        """
        self._model = model
        self._data = data
        # vectorizer
        self._vectorizer = vectorizer
        self._vectorizer_ngram_order = vectorizer_ngram_order
        self._vectorizer_analyzer = vectorizer_analyzer
        self._remove_emojis = remove_emojis
        # samples and classes
        self._samples_threshold = samples_threshold
        self._exclude_authors = exclude_authors
        self._rename_authors = rename_authors
        self._class_weight = class_weight
        self._classifier = classifier
        # evaluation
        self._eval = evaluation
        self._cv_folds = cv_folds
        self._test_fold_size = test_fold_size
        self._f1_averaging = f1_averaging

        # outcomes
        self.classes = []
        self.num_classes = 0
        self.df = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None

    def train(self):
        """
        Preprocesses data, fits a model, evaluates the classifier
        and finally saves the model to a file.
        """
        self._preprocess()
        self._build_pipeline()
        if self._eval:
            self._evaluate()
        self._fit()
        if self._eval:
            self._feature_importance()
        self._save()

    def _preprocess(self):
        """
        Reads lines from the raw Whatsapp data dump and converts
        them into a dataframe. Date and time are currently extracted
        but ignored.
        """
        d = defaultdict(list)
        previous = None, None, None

        if self._data:
            data = codecs.open(self._data, "r", "UTF-8")
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

            if author in self._exclude_authors:
                continue
            elif author in self._rename_authors:
                d[self._rename_authors[author]].append((date, time, content))
            else:
                d[author].append((date, time, content))

            previous = date, time, author

        if self._samples_threshold:
            deletes = []
            for k, v in d.iteritems():
                if len(v) < self._samples_threshold:
                    deletes.append(k)
            for k in deletes:
                del d[k]

        logging.debug("Messages with actual content:")
        for k, v in d.iteritems():
            logging.debug("%s %d" % (k, len(v)))
        logging.debug("Total messages: %d\n" %
                      sum([len(v) for v in d.values()]))

        # put in data frame
        rows = []
        index = []
        i = 0
        for k, vs in d.iteritems():
            self.classes.append(k)
            for v in vs:
                (date, time, content) = v
                rows.append({u'text': content, u'class': k})
                index.append(i)
                i += 1

        self.num_classes = len(self.classes)

        self.df = DataFrame(rows, index=index)
        logging.debug("Head of data frame before shuffling:")
        logging.debug(self.df.head())
        # shuffle for training
        self.df = self.df.reindex(numpy.random.permutation(self.df.index))
        logging.debug("Head of data frame after shuffling:")
        logging.debug(self.df.head())

    def _build_pipeline(self):
        """
        Builds an sklearn Pipeline. The pipeline consists of a kind of
        vectorizer, followed by a kind of classifier.
        """
        if self._vectorizer == "count":
            v_class = CountVectorizer
        else:
            v_class = TfidfVectorizer

        if self._remove_emojis:
            preprocessor = remove_emojis
        else:
            preprocessor = None

        self.vectorizer = v_class(sublinear_tf=True,
                                  max_df=0.5, ngram_range=(1, self._vectorizer_ngram_order),
                                  analyzer=self._vectorizer_analyzer, preprocessor=preprocessor)

        if self._classifier == "sgd-hinge":
            self.classifier = SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42,
                                            class_weight="balanced")
        else:
            raise NotImplementedError

        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("clf", self.classifier)

        ])

        logging.debug(self.vectorizer)
        logging.debug(self.classifier)
        logging.debug(self.pipeline)

    def _evaluate(self):
        """
        Performs k-fold cross validation (generalized to shuffle splits
        for arbitrary train/test ratios) and reports averaged F1 scores.

        Parts derived from:
        http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
        """
        ss = ShuffleSplit(n_splits=self._cv_folds,
                          test_size=self._test_fold_size)
        logging.debug(ss)

        scores = []
        confusion = numpy.zeros(
            (self.num_classes, self.num_classes), dtype=numpy.int)
        for train_indices, test_indices in ss.split(self.df):
            train_text = self.df.iloc[train_indices]['text'].values
            train_y = self.df.iloc[train_indices]['class'].values

            test_text = self.df.iloc[test_indices]['text'].values
            test_y = self.df.iloc[test_indices]['class'].values

            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)

            logging.info(metrics.classification_report(
                test_y, predictions, target_names=self.classes))

            confusion += confusion_matrix(test_y,
                                          predictions, labels=self.classes)
            score = f1_score(test_y, predictions, average=self._f1_averaging)
            scores.append(score)

        logging.info('Total messages classified: %d' % len(self.df))
        logging.info('Score: %f' % (sum(scores) / len(scores)))
        logging.info('Confusion matrix:')
        logging.info(confusion)

    def _feature_importance(self, k=10):
        """
        Determines the top k most informative features for each class (for
        linear classifiers).
        Source:
        http://scikit-learn.org/stable/datasets/twenty_newsgroups.html#filtering-text-for-more-realistic-training
        """
        logging.info("Top %d most informative features for each class:" % k)
        feature_names = numpy.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(self.classes):
            top_k = numpy.argsort(self.classifier.coef_[i])[-k:]
            logging.info("%s: %s" % (category, " ".join(feature_names[top_k])))

    def _fit(self):
        """
        Fits a model onto the preprocessed data.
        """
        self.pipeline.fit(self.df['text'].values, self.df['class'].values)

    def _save(self):
        """
        Save the whole pipeline to a pickled file.
        """
        from sklearn.externals import joblib
        joblib.dump(self.pipeline, self._model)
        logging.debug("Classifier saved to '%s'" % self._model)


class Predictor(object):
    """
    Predicts the author of messages, given a trained model.
    """

    def __init__(self, model):
        """
        """
        self._model = model
        self._load()

    def _load(self):
        """
        Loads a model that was previously trained and saved.
        """
        from sklearn.externals import joblib
        self.pipeline = joblib.load(self._model)
        logging.debug("Loading model pipeline from '%s'" % self._model)

    def predict(self, samples):
        """
        Predicts the class (=author) of new message samples.
        """
        predictions = []

        for sample in samples:
            sample = sample.strip()
            predictions.append((sample, self.pipeline.predict([sample])[0]))

        return predictions


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="train classifiers on Whatsapp chat data and use them for predictions")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="if --train, then save model to this path. If --predict, use saved model at this path."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        required=False,
        help="write verbose output to STDERR"
    )

    mode_options = parser.add_mutually_exclusive_group(required=True)
    mode_options.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="train a new model and save to the path -m/--model"
    )
    mode_options.add_argument(
        "--predict",
        action="store_true",
        required=False,
        help="predict classes of new samples, write predicted classes to STDOUT"
    )

    train_options = parser.add_argument_group("training parameters")

    train_options.add_argument(
        "--f1-averaging",
        type=str,
        required=False,
        choices=["micro", "macro", "weighted"],
        default="macro",
        help="if --eval, determines the type of averaging performed to compute F1 (default: macro)"
    )
    train_options.add_argument(
        "--data",
        type=str,
        required=False,
        help="path to file with raw Whatsapp data dump, UTF-8. If --data is not given, input from STDIN is assumed"
    )
    train_options.add_argument(
        "--eval",
        required=False,
        default=False,
        action="store_true",
        help="evaluate the performance on held out data and report to STDERR"
    )
    train_options.add_argument(
        "--cv-folds",
        type=int,
        required=False,
        default=5,
        metavar="K",
        help="if --eval, number of folds for cross validation (default: 5)"
    )
    train_options.add_argument(
        "--test-fold-size",
        type=float,
        required=False,
        default=0.1,
        metavar="F",
        help="if --eval, size of test fold relative to entire training set (default: 0.1)"
    )
    train_options.add_argument(
        "--vectorizer",
        type=str,
        required=False,
        choices=["count", "tfidf"],
        default="tfidf",
        help="type of vectorizer to preprocess text content (default: tfidf)"
    )
    train_options.add_argument(
        "--vectorizer-ngram-order",
        type=int,
        metavar="ORDER",
        required=False,
        default=2,
        help="vectorizer will consider ngrams in the range 1 to ORDER (default: 2)"
    )
    train_options.add_argument(
        "--vectorizer-analyzer",
        type=str,
        required=False,
        choices=["word", "char", "char_wb"],
        default="char",
        help="determines whether vectorizer features should be made of words or characters (default: char)"
    )
    train_options.add_argument(
        "--remove-emojis",
        action="store_true",
        required=False,
        default=False,
        help="strip emoji characters in high Unicode ranges from the training data (default: False)"
    )
    train_options.add_argument(
        "--samples-threshold",
        type=int,
        required=False,
        default=None,
        metavar="N",
        help="exclude classes that have fewer than N samples"
    )
    train_options.add_argument(
        "--exclude-authors",
        type=str,
        nargs="+",
        required=False,
        default=[],
        metavar="A",
        help="list names of authors that should be excluded"
    )
    train_options.add_argument(
        "--rename-authors",
        type=json.loads,
        required=False,
        default={},
        metavar="{A:R}",
        help="dict with authors that should be renamed {AUTHOR: REPLACEMENT, ...}"
    )
    train_options.add_argument(
        "--class-weight",
        action="store_true",
        required=False,
        default=False,
        help="balance uneven distribution of samples per class with weights"
    )
    train_options.add_argument(
        "--classifier",
        type=str,
        required=False,
        choices=["sgd-hinge", "mlp"],
        default="sgd-hinge",
        help="classifier to be trained (default: sgd-hinge -> SVM)"
    )

    predict_options = parser.add_argument_group("prediction parameters")

    predict_options.add_argument(
        "--output-json",
        action="store_true",
        required=False,
        help="format predicted classes as a JSON array"
    )
    predict_options.add_argument(
        "--samples",
        type=str,
        required=False,
        help="Path to file containing samples for which a class should be predicted. If --samples is not given, input from STDIN is assumed"
    )

    args = parser.parse_args()

    # avoid clash with built-in function
    args.evaluation = args.eval

    return args


def main():
    args = parse_cmd()

    # set up logging
    if args.verbose:
        level = logging.DEBUG
    elif args.evaluation:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.train:
        t = Trainer(model=args.model,
                    data=args.data,
                    vectorizer=args.vectorizer,
                    vectorizer_ngram_order=args.vectorizer_ngram_order,
                    vectorizer_analyzer=args.vectorizer_analyzer,
                    remove_emojis=args.remove_emojis,
                    samples_threshold=args.samples_threshold,
                    exclude_authors=args.exclude_authors,
                    rename_authors=args.rename_authors,
                    class_weight=args.class_weight,
                    classifier=args.classifier,
                    evaluation=args.evaluation,
                    cv_folds=args.cv_folds,
                    test_fold_size=args.test_fold_size,
                    f1_averaging=args.f1_averaging
                    )
        t.train()
    else:
        p = Predictor(model=args.model)
        if args.samples:
            input_ = codecs.open(args.samples, "r", "UTF-8")
        else:
            input_ = sys.stdin
        predictions = p.predict(samples=input_)
        for sample, prediction in predictions:
            print "%s => %s" % (sample, prediction)


if __name__ == '__main__':
    main()
