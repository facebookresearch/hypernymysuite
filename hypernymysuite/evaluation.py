#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.stats
from sklearn.metrics import average_precision_score, precision_recall_curve

import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer


# Lemmatizer we need for the model
lemmatizer = WordNetLemmatizer()

# Datasets!
CORRELATION_EVAL_DATASETS = [("hyperlex", "data/hyperlex_rnd.tsv")]

SIEGE_EVALUATIONS = [
    ("bless", "data/bless.tsv"),
    ("leds", "data/leds.tsv"),
    ("eval", "data/eval.tsv"),
    ("weeds", "data/wbless.tsv"),
    ("shwartz", "data/shwartz.tsv"),
]


class Dataset(object):
    """
    Represents a hypernymy dataset, which contains a left hand side (LHS) of hyponyms,
    and right hand side (RHS) of hypernyms.

    Params:
        filename: str. Filename on disk corresponding to the TSV file
        vocabdict: dict[str,*]. Dictionary whose keys are the vocabulary of the
            model to test
        ycolumn: str. Optional name of the label column.
    """

    def __init__(self, filename, vocabdict, ycolumn="label"):
        if "<OOV>" not in vocabdict:
            raise ValueError("Reserved word <OOV> must appear in vocabulary.")

        table = pd.read_table(filename)

        # some things require the part of speech, which may not be explicitly
        # given in the dataset.
        if "pos" not in table.columns:
            table["pos"] = "N"
        table = table[table.pos.str.lower() == "n"]

        # Handle MWEs by replacing the space
        table["word1"] = table.word1.apply(lambda x: x.replace(" ", "_").lower())
        table["word2"] = table.word2.apply(lambda x: x.replace(" ", "_").lower())

        if vocabdict:
            self.word1_inv = table.word1.apply(vocabdict.__contains__)
            self.word2_inv = table.word2.apply(vocabdict.__contains__)
        else:
            self.word1_inv = table.word1.apply(lambda x: True)
            self.word2_inv = table.word2.apply(lambda x: True)

        # Always evaluate on lemmas
        table["word1"] = table.word1.apply(lemmatizer.lemmatize)
        table["word2"] = table.word2.apply(lemmatizer.lemmatize)

        self.table = table
        self.labels = np.array(table[ycolumn])
        if "fold" in table:
            self.folds = table["fold"]
        else:
            self.folds = np.array(["test"] * len(self.table))

        self.table["is_oov"] = self.oov_mask

    def __len__(self):
        return len(self.table)

    @property
    def hypos(self):
        return np.array(self.table.word1)

    @property
    def hypers(self):
        return np.array(self.table.word2)

    @property
    def invocab_mask(self):
        return self.word1_inv & self.word2_inv

    @property
    def oov_mask(self):
        return ~self.invocab_mask

    @property
    def val_mask(self):
        return np.array(self.folds == "val")

    @property
    def test_mask(self):
        return np.array(self.folds == "test")

    @property
    def train_mask(self):
        return np.array(self.folds == "train")

    @property
    def train_inv_mask(self):
        return self.invocab_mask & self.train_mask

    @property
    def val_inv_mask(self):
        return self.invocab_mask & self.val_mask

    @property
    def test_inv_mask(self):
        return self.invocab_mask & self.test_mask

    @property
    def y(self):
        return self.labels


def correlation_setup(filename, model):
    """
    Computes a spearman's rho correlation between model and continuous value.
    """
    ds = Dataset(filename, model.vocab, ycolumn="score")

    h = model.predict_many(ds.hypos, ds.hypers)
    # For OOV words, we should guess the median distance of all the pairs.
    # i.e. We're not committing to high or low similarity
    h[ds.oov_mask] = np.median(h[ds.train_inv_mask])

    y = ds.labels
    mi = ds.invocab_mask
    m_train = ds.train_mask
    mi_train = ds.train_inv_mask
    m_val = ds.val_mask
    mi_val = ds.val_inv_mask
    m_test = ds.test_mask
    mi_test = ds.val_inv_mask

    return {
        "rho_train": scipy.stats.spearmanr(y[m_train], h[m_train])[0],
        "rho_val": scipy.stats.spearmanr(y[m_val], h[m_val])[0],
        "rho_test": scipy.stats.spearmanr(y[m_test], h[m_test])[0],
        "rho_all": scipy.stats.spearmanr(y, h)[0],
        "rho_train_inv": scipy.stats.spearmanr(y[mi_train], h[mi_train])[0],
        "rho_val_inv": scipy.stats.spearmanr(y[mi_val], h[mi_val])[0],
        "rho_test_inv": scipy.stats.spearmanr(y[mi_test], h[mi_test])[0],
        "rho_all_inv": scipy.stats.spearmanr(y[mi], h[mi])[0],
        "num_all": len(ds),
        "num_oov_all": int(sum(ds.oov_mask)),
        "pct_oov_all": np.mean(ds.oov_mask),
    }


def bless_directionality_setup(model):
    """
    Asks a model whether (x, y) > (y, x) for a number of positive hypernymy examples.
    """
    # load up the data
    ds = Dataset("data/bless.tsv", model.vocab)

    # only keep the positive pairs
    hypos = ds.hypos[ds.y]
    hypers = ds.hypers[ds.y]

    forward_predictions = model.predict_many(hypos, hypers)
    reverse_predictions = model.predict_many(hypers, hypos)

    # Fold masks
    m_val = ds.val_mask[ds.y]
    m_test = ds.test_mask[ds.y]

    # Fold, in-vocab masks
    oov = ds.oov_mask[ds.y]
    mi_val = m_val & ~oov
    mi_test = m_test & ~oov

    # final input
    # Check that the original directionality is correct
    yhat = forward_predictions > reverse_predictions

    return {
        "acc_val": np.mean(yhat[m_val]),
        "acc_test": np.mean(yhat[m_test]),
        "acc_all": np.mean(yhat),
        "acc_val_inv": np.mean(yhat[mi_val]),
        "acc_test_inv": np.mean(yhat[mi_test]),
        "acc_all_inv": np.mean(yhat[~oov]),
        "num_val": int(sum(m_val)),
        "num_test": int(sum(m_test)),
        "num_oov_all": int(sum(oov)),
        "pct_oov_all": np.mean(oov),
    }


def wbless_setup(model):
    """
    Accuracy using a threshold, with a dataset that explicitly contains reverse pairs.
    """
    ds = Dataset("data/wbless.tsv", model.vocab)

    # Ensure we always get the same results
    rng = np.random.RandomState(42)
    VAL_PROB = .02
    NUM_TRIALS = 1000

    # We have no way of handling oov
    h = model.predict_many(ds.hypos[ds.invocab_mask], ds.hypers[ds.invocab_mask])
    y = ds.y[ds.invocab_mask]

    val_scores = []
    test_scores = []

    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
        # Test is everything except val
        m_test = ~m_val
        _, _, t = precision_recall_curve(y[m_val], h[m_val])
        # pick the highest accuracy on the validation set
        thr_accs = np.mean((h[m_val, np.newaxis] >= t) == y[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]
        preds_val = h[m_val] >= best_t
        preds_test = h[m_test] >= best_t
        # Evaluate
        val_scores.append(np.mean(preds_val == y[m_val]))
        test_scores.append(np.mean(preds_test == y[m_test]))
        # sanity check
        assert np.allclose(val_scores[-1], thr_accs.max())

    # report average across many folds
    return {"acc_val_inv": np.mean(val_scores), "acc_test_inv": np.mean(test_scores)}


def bibless_setup(model):
    """
    Combined detection with a threshold, plus direction prediction.
    """
    ds = Dataset("data/bibless.tsv", model.vocab)

    # Ensure we always get the same results
    rng = np.random.RandomState(42)
    VAL_PROB = .02
    NUM_TRIALS = 1000

    # We have no way of handling oov
    y = ds.y[ds.invocab_mask]

    # hypernymy could be either direction
    yh = y != 0

    # get forward and backward predictions
    hf = model.predict_many(ds.hypos[ds.invocab_mask], ds.hypers[ds.invocab_mask])
    hr = model.predict_many(ds.hypers[ds.invocab_mask], ds.hypos[ds.invocab_mask])
    h = np.max([hf, hr], axis=0)

    dir_pred = 2 * np.float32(hf >= hr) - 1

    val_scores = []
    test_scores = []
    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
        # Test is everything except val
        m_test = ~m_val

        # set the threshold based on the maximum score
        _, _, t = precision_recall_curve(yh[m_val], h[m_val])
        thr_accs = np.mean((h[m_val, np.newaxis] >= t) == yh[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]

        det_preds_val = h[m_val] >= best_t
        det_preds_test = h[m_test] >= best_t

        fin_preds_val = det_preds_val * dir_pred[m_val]
        fin_preds_test = det_preds_test * dir_pred[m_test]

        val_scores.append(np.mean(fin_preds_val == y[m_val]))
        test_scores.append(np.mean(fin_preds_test == y[m_test]))

    # report average across many folds
    return {"acc_val_inv": np.mean(val_scores), "acc_test_inv": np.mean(test_scores)}


def ap_at_k(y_true, y_score, k):
    """
    Computes AP@k, or AP of the model's top K predictions. Used in
    Shwartz, Santus and Schlectweg, EACL 2017.
    https://arxiv.org/abs/1612.04460
    """
    argsort = np.argsort(y_score)
    score_srt = y_score[argsort[-k:]]
    label_srt = y_true[argsort[-k:]]
    return average_precision_score(label_srt, score_srt)


def siege_setup(filename, model):
    """
    Computes Average Precision for a binary dataset.
    """
    ds = Dataset(filename, model.vocab)

    m_val = ds.val_mask
    mi_val = ds.val_inv_mask
    m_test = ds.test_mask
    mi_test = ds.test_inv_mask

    # we only need to compute forward on in-vocab words, speeds things up
    h_inv = model.predict_many(ds.hypos[ds.invocab_mask], ds.hypers[ds.invocab_mask])

    # stub out for the entire data though
    h = np.zeros(len(ds))
    # and fill with our predictions
    h[ds.invocab_mask] = h_inv
    # And OOV predictions should be our lowest validation prediction, essentially
    # always predicting false
    h[ds.oov_mask] = h[mi_val].min()
    y = ds.y
    results = {}

    # Across all relations
    results["other"] = {
        "ap_val": average_precision_score(y[m_val], h[m_val]),
        "ap_test": average_precision_score(y[m_test], h[m_test]),
        "ap100_val": ap_at_k(y[m_val], h[m_val], 100),
        "ap100_test": ap_at_k(y[m_test], h[m_test], 100),
        "ap_val_inv": average_precision_score(y[mi_val], h[mi_val]),
        "ap_test_inv": average_precision_score(y[mi_test], h[mi_test]),
        "ap100_val_inv": ap_at_k(y[mi_val], h[mi_val], 100),
        "ap100_test_inv": ap_at_k(y[mi_test], h[mi_test], 100),
    }
    results["pct_oov"] = np.mean(ds.oov_mask)
    return results


def all_evaluations(model, extra_args=None):
    """
    Utility method which performs all evaluations and unifies the results into
    a single (nested) dictionary.

    Args:
        model: HypernymySuiteModel. Model to be evaluated.

    Returns:
        A nested dictionary of results. Best combined with `compile_table`
            module.
    """
    results = {}
    results["dir_wbless"] = wbless_setup(model)
    results["dir_bibless"] = bibless_setup(model)
    results["dir_dbless"] = bless_directionality_setup(model)

    for taskname, filename in CORRELATION_EVAL_DATASETS:
        result = correlation_setup(filename, model)
        results["cor_{}".format(taskname)] = result

    # siege evaluations
    for taskname, filename in SIEGE_EVALUATIONS:
        result = siege_setup(filename, model)
        results["siege_{}".format(taskname)] = result

    return results
