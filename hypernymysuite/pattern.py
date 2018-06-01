#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Hearst pattern based models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sparse

from .base import HypernymySuiteModel
from .reader import read_sparse_matrix


class PatternBasedModel(HypernymySuiteModel):
    """
    Basis class for all Hearst-pattern based approaches.
    """

    def __init__(self, patterns_filename):
        csr_m, self.id2word, self.vocab, _ = read_sparse_matrix(
            patterns_filename, same_vocab=True
        )
        self.p_w = csr_m.sum(axis=1).A[:, 0]
        self.p_c = csr_m.sum(axis=0).A[0, :]
        self.matrix = csr_m.todok()
        self.N = self.p_w.sum()

    def predict(self, hypo, hyper):
        raise NotImplementedError("Abstract class")

    def __str__(self):
        raise NotImplementedError("Abstract class")


class RawCountModel(PatternBasedModel):
    """
    P(x, y) model which uses raw counts.
    """

    def predict(self, hypo, hyper):
        L = self.vocab.get(hypo, 0)
        R = self.vocab.get(hyper, 0)
        return self.matrix[(L, R)]

    def __str__(self):
        return "raw"


class PPMIModel(RawCountModel):
    """
    PPMI(x, y) model which uses a PPMI-transformed Hearst patterns for
    predictions.
    """

    def __init__(self, patterns_filename):
        # first read in the normal stuff
        super(PPMIModel, self).__init__(patterns_filename)
        # now let's transform the matrix
        tr_matrix = sparse.dok_matrix(self.matrix.shape)
        # actually do the transformation
        for (l, r) in self.matrix.keys():
            pmi_lr = (
                np.log(self.N)
                + np.log(self.matrix[(l, r)])
                - np.log(self.p_w[l])
                - np.log(self.p_c[r])
            )
            # ensure it's /positive/ pmi
            ppmi_lr = np.clip(pmi_lr, 0.0, 1e12)
            tr_matrix[(l, r)] = ppmi_lr
        self.matrix = tr_matrix

    def __str__(self):
        return "ppmi"


class _SvdMixIn(object):
    """
    Abstract mixin, do not use directly. Computes the SVD on top of the matrix
    from the superclass (may only be mixed in with a PatternBasedModel).
    """

    def __init__(self, pattern_filename, k):
        # First make sure the matrix is loaded
        super(_SvdMixIn, self).__init__(pattern_filename)
        self.k = k
        U, S, V = sparse.linalg.svds(self.matrix.tocsr(), k=k)
        self.U = U.dot(np.diag(S))
        self.V = V.T

    def predict(self, hypo, hyper):
        L = self.vocab.get(hypo, 0)
        R = self.vocab.get(hyper, 0)
        return self.U[L].dot(self.V[R])

    def predict_many(self, hypos, hypers):
        lhs = [self.vocab.get(x, 0) for x in hypos]
        rhs = [self.vocab.get(x, 0) for x in hypers]

        retval = np.sum(self.U[lhs] * self.V[rhs], axis=1)
        return retval

    def __str__(self):
        return "svd" + super(_SvdMixIn, self).__str__()


class SvdRawModel(_SvdMixIn, RawCountModel):
    """
    sp(x,y) model presented in the paper. This is an svd over the raw counts.
    """

    pass


class SvdPpmiModel(_SvdMixIn, PPMIModel):
    """
    spmi(x,y) model presented in the paper. This is the svd over the ppmi matrix.
    """

    pass


class RandomBaseline(PatternBasedModel):
    """
    Generates random, but consistent predictions. Essentially a random
    matrix factorization.
    """

    def __init__(self, filename, k=10, seed=42):
        super(RandomBaseline, self).__init__(filename)
        np.random.seed(seed)
        self.L = np.random.rand(len(self.vocab), k)
        self.R = np.random.rand(len(self.vocab), k)

    def predict(self, hypo, hyper):
        lid = self.vocab.get(hypo, 0)
        rid = self.vocab.get(hyper, 0)
        return self.L[lid].dot(self.R[rid])

    def __str__(self):
        return "random"
