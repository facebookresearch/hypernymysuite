#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility module for easily reading a sparse matrix.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
import os
import gzip
import scipy.sparse as sp

try:
    import cPickle as pickle
except ImportError:
    import pickle


def __try_three_columns(string):
    fields = string.split("\t")
    if len(fields) > 3:
        fields = fields[:3]
    if len(fields) == 3:
        return fields[0], fields[1], float(fields[2])
    if len(fields) == 2:
        return fields[0], fields[1], 1.0
    else:
        raise ValueError("Invalid number of fields {}".format(len(fields)))


def __load_sparse_matrix(filename, same_vocab):
    """
    Actual workhorse for loading a sparse matrix. See docstring for
    read_sparse_matrix.

    """
    objects = ["<OOV>"]
    rowvocab = {"<OOV>": 0}
    if same_vocab:
        colvocab = rowvocab
    else:
        colvocab = {}
    _is = []
    _js = []
    _vs = []

    # Read gzip files
    if filename.endswith(".gz"):
        f = gzip.open(filename, "r")
    else:
        f = open(filename, "rb")

    for line in f:
        line = line.decode("utf-8")
        target, context, weight = __try_three_columns(line)
        if target not in rowvocab:
            rowvocab[target] = len(rowvocab)
            objects.append(target)
        if context not in colvocab:
            colvocab[context] = len(colvocab)
            if same_vocab:
                objects.append(context)

        _is.append(rowvocab[target])
        _js.append(colvocab[context])
        _vs.append(weight)

    # clean up
    f.close()

    _shape = (len(rowvocab), len(colvocab))
    spmatrix = sp.csr_matrix((_vs, (_is, _js)), shape=_shape, dtype=np.float64)
    return spmatrix, objects, rowvocab, colvocab


def read_sparse_matrix(filename, allow_binary_cache=False, same_vocab=False):
    """
    Reads in a 3 column file as a sparse matrix, where each line (x, y, v)
    gives the name of the row x, column y, and the value z.

    If filename ends with .gz, will assume the file is gzip compressed.

    Args:
        filename: str. The filename containing sparse matrix in 3-col format.
        allow_binary_cache: bool. If true, caches the matrix in a pkl file with
            the same filename for faster reads. If cache doesn't exist, will
            create it.
        same_vocab: bool. Indicates whether rows and columns have the same vocab.

    Returns:
        A tuple containing (spmatrix, id2row, row2id, col2id):
            spmatrix: a scipy.sparse matrix with the entries
            id2row: a list[str] containing the names for the rows of the matrix
            row2id: a dict[str,int] mapping words to row indices
            col2id: a dict[str,int] mapping words to col indices. If same_vocab,
                this is identical to row2id.
    """
    # make sure the cache is new enough
    cache_filename = filename + ".pkl"
    cache_exists = os.path.exists(cache_filename)
    cache_fresh = cache_exists and os.path.getmtime(filename) <= os.path.getmtime(
        cache_filename
    )
    if allow_binary_cache and cache_fresh:
        logging.debug("Using space cache {}".format(cache_filename))
        with open(cache_filename + ".pkl", "rb") as pklf:
            return pickle.load(pklf)
    else:
        # binary cache is not allowed, or it's stale
        result = __load_sparse_matrix(filename, same_vocab=same_vocab)
        if allow_binary_cache:
            logging.warning("Dumping the binary cache {}.pkl".format(filename))
            with open(filename + ".pkl", "wb") as pklf:
                pickle.dump(result, pklf)
        return result
