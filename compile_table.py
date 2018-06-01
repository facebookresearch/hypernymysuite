#!/usr/bin/env python3

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
import pandas as pd
import numpy as np
import json

logging.basicConfig(level=logging.INFO)

REPORT_FIELDS = [
    ("Siege", "BLESS", "siege_bless_other_ap_{fold}"),
    ("Siege", "EVAL", "siege_eval_other_ap_{fold}"),
    ("Siege", "LEDS", "siege_leds_other_ap_{fold}"),
    ("Siege", "Shwartz", "siege_shwartz_other_ap_{fold}"),
    ("Siege", "WBless", "siege_weeds_other_ap_{fold}"),
    ("Graded", "Hyperlex", "cor_hyperlex_rho_{fold}"),
    ("Direction", "BLESS", "dir_dbless_acc_{fold}"),
    ("Direction", "Wbless", "dir_wbless_acc_{fold}_inv"),
    ("Direction", "BiBless", "dir_bibless_acc_{fold}_inv"),
]

order = {}
for i, (_, metric, _) in enumerate(REPORT_FIELDS):
    if metric not in order:
        order[metric] = i


def nice_grouping(df):
    r = df.pivot_table(index=["modeltype"], columns=["metric"], values="score")
    cols = sorted(r.columns, key=lambda x: order[x])
    return r[cols]


def fprint(x):
    if np.isnan(x):
        return ""
    if x <= 1:
        return ("%.2f" % x).replace("0.", ".")
    else:
        return "%d" % x


def gather_metrics(results):
    """
    Gathers up metrics across all the different report fields, and puts them into
    one nice groupable table.
    """
    output = []
    for modeltype, modelset in results.groupby("name"):
        for tablename, metricname, metrickey in REPORT_FIELDS:
            # gather up the best score by validation fold of this model group
            modelset = modelset.copy().reset_index(drop=True)
            valkey = metrickey.replace("{fold}", "val")
            testkey = metrickey.replace("{fold}", "test")

            if valkey not in results.columns:
                continue

            if valkey not in modelset.columns:
                modelset[valkey] = np.nan
            if testkey not in modelset.columns:
                modelset[testkey] = np.nan

            # find the best model on validation
            modelset = modelset.sort_values(valkey, ascending=(tablename == "Ranking"))
            best_on_val = modelset.head(1).iloc[0]
            val_score = best_on_val[valkey]
            test_score = best_on_val[testkey]

            # report results
            output.append(
                {
                    "modeltype": modeltype,
                    "tablename": tablename,
                    "metric": metricname,
                    "fold": "val",
                    "score": val_score,
                }
            )
            output.append(
                {
                    "modeltype": modeltype,
                    "tablename": tablename,
                    "metric": metricname,
                    "fold": "test",
                    "score": test_score,
                }
            )

    df = pd.DataFrame(output)
    return df


def output_latex(nice_subset):
    return nice_subset.to_latex(sys.stdout, float_format=fprint)


def output_html(nice_subset):
    return nice_subset.to_html(
        sys.stdout, col_space=100, border=0, float_format=fprint, justify="right"
    )


def output_console(subset):
    return subset.to_string(sys.stdout, float_format=fprint, justify="right")


def __flatten_dict(d, joiner="_"):
    items = []
    for k, v in d.items():
        if type(v) is dict:
            for k2, v2 in __flatten_dict(v, joiner=joiner):
                items.append((k + joiner + k2, v2))
        else:
            items.append((k, v))
    return items


def read_json_log(filename):
    """
    Reads a json log as output from a given model run. Does simple filtering
    to prevent bad plots (e.g. drop NaN lines), and flattens the dictionary.
    """
    output = []
    if filename == "-" or "":
        f = sys.stdin
    else:
        f = open(filename)  # noqa: P201

    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        line = line.replace("NaN", "null")
        line = line.replace("-Infinity", "null")
        line = line.replace("Infinity", "null")
        try:
            d = json.loads(line)
            output.append(dict(__flatten_dict(d)))
        except ValueError:
            logging.warning("Warning: Line {} of {} didn't parse: ".format(i, filename))

    f.close()
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description="Compiles results into a table.")
    parser.add_argument("--input", "-i", default="-", help="Input logs")
    parser.add_argument("--latex", action="store_true", help="Output latex")
    parser.add_argument("--html", action="store_true", help="Output html")
    parser.add_argument("--test", action="store_true", help="Display test results")
    args = parser.parse_args()

    # Rad in the log format
    results = read_json_log(args.input)
    # For output, we want to limit the number of decimal points and auto round
    pd.set_option("precision", 2)
    # And for the purpose of output, don't allow wrapping
    pd.set_option("display.width", 100000)

    # baselines replace "distfn" with the baseline name for simplicitly of grouping
    df = gather_metrics(results)
    for tablename, subset in df.groupby("tablename"):
        subset = subset.copy().reset_index(drop=False)
        nice_val = nice_grouping(subset[subset.fold == "val"])
        nice_test = nice_grouping(subset[subset.fold == "test"])
        if args.test:
            nice_subset = nice_test
        else:
            nice_subset = nice_val

        if args.latex:
            output_latex(nice_subset)
        elif args.html:
            sys.stdout.write(
                "<style type='text/css'>"
                "td { padding: 0.1em; text-align: right; }"
                "</style>\n"
            )
            output_html(nice_subset)
        else:
            print(tablename)
            output_console(nice_subset)
            print()
        print()


if __name__ == "__main__":
    main()
