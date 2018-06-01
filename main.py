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

import sys
import argparse
import logging
import json

from hypernymysuite import base
from hypernymysuite import pattern
from hypernymysuite import evaluation
from hypernymysuite import unsup


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="Baseline to run")
    parser.add_argument("--dset", help="Corpus of hearst patterns.")
    parser.add_argument("--k", default=None, type=int, help="Number of dimensions.")
    args = parser.parse_args()

    if args.cmd in {"svdppmi", "svdcnt", "random", "slqs", "slqscos"} and not args.k:
        raise parser.error("You must specify --k")

    if args.cmd == "cnt":
        model = pattern.RawCountModel(args.dset)
    elif args.cmd == "ppmi":
        model = pattern.PPMIModel(args.dset)
    elif args.cmd == "svdppmi":
        model = pattern.SvdPpmiModel(args.dset, k=args.k)
    elif args.cmd == "svdcnt":
        model = pattern.SvdRawModel(args.dset, k=args.k)
    elif args.cmd == "random":
        model = base.RandomBaseline(args.dset, k=args.k)
    elif args.cmd == "weeds":
        model = unsup.UnsupervisedBaseline(args.dset, unsup.weeds_prec)
    elif args.cmd == "invcl":
        model = unsup.UnsupervisedBaseline(args.dset, unsup.invCL)
    elif args.cmd == "slqs":
        model = unsup.SLQS(args.dset, args.k)
    elif args.cmd == "slqscos":
        model = unsup.SLQS_Cos(args.dset, args.k)
    elif args.cmd == "cosine":
        model = unsup.UnsupervisedBaseline(args.dset, unsup.cosine)
    elif args.cmd == "precomputed":
        model = unsup.Precomputed(args.dset)
    else:
        parser.print_help()
        sys.exit(1)

    result = evaluation.all_evaluations(model, args)
    result["name"] = args.cmd
    result["dset"] = args.dset
    result["k"] = args.k
    print(json.dumps(result))


if __name__ == "__main__":
    main()
