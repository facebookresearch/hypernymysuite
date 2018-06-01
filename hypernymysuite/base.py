#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract class module which defines the interface for our HypernymySuite model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class HypernymySuiteModel(object):
    """
    Base class for all hypernymy suite models.

    To use this, must implement these methods:

        predict(self, hypo: str, hyper: str): float, which makes a
            prediction about two words.
        vocab: dict[str, int], which tells if a word is in the
            vocabulary.

    Your predict method *must* be prepared to handle OOV terms, but it may
    returning any sentinel value you wish.

    You can optionally implement
        predict_many(hypo: list[str], hyper: list[str]: array[float]

    The skeleton method here will just call predict() in a for loop, but
    some methods can be vectorized for improved performance. This is the
    actual method called by the evaluation script.
    """

    vocab = {}

    def __init__(self):
        raise NotImplementedError

    def predict(self, hypo, hyper):
        """
        Core modeling procedure, estimating the degree to which hypo is_a hyper.

        This is an abstract method, describing the interface.

        Args:
            hypo: str. A hypothesized hyponym.
            hyper: str. A hypothesized hypernym.

        Returns:
            float. The score estimating the degree to which hypo is_a hyper.
                Higher values indicate a stronger degree.
        """
        raise NotImplementedError

    def predict_many(self, hypos, hypers):
        """
        Make predictions for many pairs at the same time. The default
        implementation just calls predict() many times, but many models
        benefit from vectorization.

        Args:
            hypos: list[str]. A list of hypothesized hyponyms.
            hypers: list[str]. A list of corresponding hypothesized hypernyms.
        """
        result = []
        for x, y in zip(hypos, hypers):
            result.append(self.predict(x, y))
        return np.array(result, dtype=np.float32)


class Precomputed(HypernymySuiteModel):
    """
    A model which uses precomputed prediction, read from a TSV file.
    """

    def __init__(self, precomputed):
        self.vocab = {"<OOV>": 0}
        self.lookup = {}
        with open(precomputed) as f:
            for line in f:
                w1, w2, sim, is_oov = line.strip().split("\t")
                if w1 == "hypo" and w2 == "hyper":
                    # header, ignore it
                    continue
                if is_oov == "1" or is_oov.lower() in ("t", "true"):
                    # Don't read in oov predictions
                    continue
                if w1 not in self.vocab:
                    self.vocab[w1] = len(self.vocab)
                if w2 not in self.vocab:
                    self.vocab[w2] = len(self.vocab)
                sim = float(sim)
                self.lookup[(self.vocab[w1], self.vocab[w2])] = sim

    def predict(self, hypo, hyper):
        x = self.vocab.get(hypo, 0)
        y = self.vocab.get(hyper, 0)
        return self.lookup.get((x, y), 0.)
