#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# -------------------------------------------------------------------------------
# This shell script generates the table presented in the paper
# -------------------------------------------------------------------------------

# Quit if there's any errors
set -e

# There must be exactly one argument
if [[ "$#" != "1" ]]
then
    echo "Usage: $0 output_filename.json" >&2
    echo "You must provide an output filename to store results." >&2
    exit 1
fi

RESULTS_FILE="$1"
DSET="hearst_counts.txt.gz"

if [[ -f "$RESULTS_FILE" ]]
then
    echo "$RESULTS_FILE already exists. Refusing to overwrite it." >&2
    echo "Run 'rm $RESULTS_FILE' if this is intentional." >&2
    exit 1
fi

echo "Computing results. Please be very patient." >&2

# p(x,y) model
python main.py cnt --dset "$DSET" > "$RESULTS_FILE"
# ppmi(x,y) model
python main.py ppmi --dset "$DSET" >> "$RESULTS_FILE"

for k in 5 10 15 20 25 50 100 150 200 250 300 500 1000
do
    # sp(x, y) model
    python main.py svdcnt --dset "$DSET" --k "$k" >> "$RESULTS_FILE"
    # spmi(x, y) model
    python main.py svdppmi --dset "$DSET" --k "$k" >> "$RESULTS_FILE"
done

# Finally present results
echo "Validation performance:"
python compile_table.py -i "$RESULTS_FILE"

echo
echo

echo "Test performance:"
python compile_table.py -i "$RESULTS_FILE" --test
