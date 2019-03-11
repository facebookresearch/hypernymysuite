#!/bin/bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# -------------------------------------------------------------------------------
# This shell script downloads and preprocesses all the datasets
# -------------------------------------------------------------------------------

# Immediately quit on error
set -e

# if you have any proxies, etc., put them here
CURL_OPTIONS="-s"


# URLS of each of the different datasets
OMER_URL="http://u.cs.biu.ac.il/~nlp/wp-content/uploads/lexical_inference.zip"
SHWARTZ_URL="https://raw.githubusercontent.com/vered1986/HypeNET/v2/dataset/datasets.rar"
VERED_REPO_URL="https://raw.githubusercontent.com/vered1986/UnsupervisedHypernymy/e3b22709365c7b3042126e5887c9baa03631354e/datasets"
KIMANH_REPO_URL="https://raw.githubusercontent.com/nguyenkh/HyperVec/bd2cb15a6be2a4726ffbf9c0d7e742144790dee3/datasets_classification"
HYPERLEX_URL="http://people.ds.cam.ac.uk/iv250/paper/hyperlex/hyperlex-data.zip"

function warning () {
    echo "$1" >&2
}

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

function deterministic_shuffle () {
    # sort randomly but with a predictable seed
    sort --random-sort --random-source=<(get_seeded_random 42)
}

function download_hyperlex () {
    TMPFILE="$(mktemp)"
    TMPDIRE="$(mktemp -d)"
    curl $CURL_OPTIONS "$HYPERLEX_URL" > "$TMPFILE"
    unzip "$TMPFILE" -d "$TMPDIRE" > /dev/null

    echo -e 'word1\tword2\tpos\tlabel\tscore\tfold'
    grep -v WORD1 "$TMPDIRE/splits/random/hyperlex_training_all_random.txt" | \
        cut -d' ' -f1-5 | tr ' ' '\t' | \
        awk -F'\t' '$0=$0"\ttrain"'
    grep -v WORD1 "$TMPDIRE/splits/random/hyperlex_dev_all_random.txt" | \
        cut -d' ' -f1-5 | tr ' ' '\t' | \
        awk -F'\t' '$0=$0"\tval"'
    grep -v WORD1 "$TMPDIRE/splits/random/hyperlex_test_all_random.txt" | \
        cut -d' ' -f1-5 | tr ' ' '\t' | \
        awk -F'\t' '$0=$0"\ttest"'

    rm -rf "$TMPFILE" "$TMPDIRE"
}

function download_bless () {
    TMPFILE="$(mktemp)"
    TMPDIRE="$(mktemp -d)"
    curl $CURL_OPTIONS "$OMER_URL" > "$TMPFILE"
    unzip "$TMPFILE" -d "$TMPDIRE" > /dev/null

    echo -e 'word1\tword2\tlabel\tfold'
    cat "${TMPDIRE}/lexical_entailment/bless2011/data_rnd_test.tsv" \
        "${TMPDIRE}/lexical_entailment/bless2011/data_rnd_train.tsv" \
        "${TMPDIRE}/lexical_entailment/bless2011/data_rnd_val.tsv" | \
        tr -d '\15' | \
        deterministic_shuffle | \
        awk '{if (NR < 1454) {print $0 "\tval"} else {print $0 "\ttest"}}'

    rm -rf "$TMPFILE" "$TMPDIRE"
}

function download_leds () {
    TMPFILE="$(mktemp)"
    TMPDIRE="$(mktemp -d)"
    curl $CURL_OPTIONS "$OMER_URL" > "$TMPFILE"
    unzip "$TMPFILE" -d "$TMPDIRE" > /dev/null

    echo -e 'word1\tword2\tlabel\tfold'
    cat "${TMPDIRE}/lexical_entailment/baroni2012/data_rnd_test.tsv" \
        "${TMPDIRE}/lexical_entailment/baroni2012/data_rnd_train.tsv" \
        "${TMPDIRE}/lexical_entailment/baroni2012/data_rnd_val.tsv" | \
        tr -d '\15' | \
        deterministic_shuffle | \
        awk '{if (NR < 276) {print $0 "\tval"} else {print $0 "\ttest"}}'

    rm -rf "$TMPFILE" "$TMPDIRE"
}

function download_shwartz () {
    TMPFILE="$(mktemp)"
    TMPDIRE="$(mktemp -d)"
    curl $CURL_OPTIONS "$SHWARTZ_URL" > "$TMPFILE"

    unrar x "$TMPFILE" "$TMPDIRE" >/dev/null
    echo -e 'word1\tword2\tlabel\tfold'
    cat "$TMPDIRE/dataset_rnd/train.tsv" \
        "$TMPDIRE/dataset_rnd/test.tsv" \
        "$TMPDIRE/dataset_rnd/val.tsv" | \
        grep -v ' ' | \
        deterministic_shuffle | \
        awk '{if (NR < 5257) {print $0 "\tval"} else {print $0 "\ttest"}}'

    rm -rf "$TMPFILE" "$TMPDIRE"
}

function download_bibless () {
    echo -e 'word1\tword2\trelation\tlabel'
    curl $CURL_OPTIONS "$KIMANH_REPO_URL/ABIBLESS.txt" | \
        cut -f1,2,4 | \
        awk -F'\t' '{if ($3 == "hyper") {print $0 "\t1"} else if ($3 == "other") {print $0 "\t0"} else {print $0 "\t-1"}}'
}

function download_wbless () {
    echo -e 'word1\tword2\tlabel\trelation\tfold'
    curl $CURL_OPTIONS "$KIMANH_REPO_URL/AWBLESS.txt" | \
        deterministic_shuffle | \
        awk '{if (NR < 168) {print $0 "\tval"} else {print $0 "\ttest"}}'
}

function download_eval () {
    echo -e 'word1\tword2\tlabel\trelation\tfold'
    curl $CURL_OPTIONS "$VERED_REPO_URL/EVALution.val" "$VERED_REPO_URL/EVALution.test" | \
        sed 's/-[jvn]\t/\t/g' | sort | uniq | \
        deterministic_shuffle | \
        awk '{if (NR < 737) {print $0 "\tval"} else {print $0 "\ttest"}}'
}



OUTPUT="data"

if [ -d "$OUTPUT" ]
then
    echo "Warning: Already found the data. Please run 'rm -rf $OUTPUT'" >&2
    exit 1
fi

if [ ! -x "$(command -v unrar)" ]
then
    warning "This script requires the 'unrar' tool. Please run"
    warning "  brew install unrar"
    warning "or whatever your system's equivalent is."
    exit 1
fi

if [ ! -x "$(command -v openssl)" ]
then
    warning "This script requires the 'openssl' tool. Please run"
    warning "  brew install unrar"
    warning "or whatever your system's equivalent is."
    exit 1
fi



# prep the output folder
mkdir -p "$OUTPUT"


warning "[1/7] Downloading BLESS"
download_bless > "$OUTPUT/bless.tsv"

warning "[2/7] Downloading LEDS"
download_leds > "$OUTPUT/leds.tsv"

warning "[3/7] Downloading EVAL"
download_eval > "$OUTPUT/eval.tsv"

warning "[4/7] Downloading Shwartz"
download_shwartz > "$OUTPUT/shwartz.tsv"

warning "[5/7] Downloading Hyperlex"
download_hyperlex > "$OUTPUT/hyperlex_rnd.tsv"

warning "[6/7] Downloading WBLESS"
download_wbless > "$OUTPUT/wbless.tsv"

warning "[7/7] Downloading BiBLESS"
download_bibless > "$OUTPUT/bibless.tsv"

warning "All done."
