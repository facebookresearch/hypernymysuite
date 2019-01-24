# Hypernymy Suite

HypernymySuite is a tool for evaluating some hypernymy detection modules. Its
predominant focus is reproducing the results for the following paper.

> Stephen Roller, Douwe Kiela, and Maximilian Nickel. 2018. Hearst Patterns
> Revisited: Automatic Hypernym Detection from Large Text Corpora. ACL.
> ([arXiv](https://arxiv.org/abs/1806.03191))

We hope that open sourcing our evaluation will help facilitate future research.

## Example

Before you begin, you should run the script to download the evaluation datasets.

    bash download_data.sh

You can produce results in a JSON format by calling main.py:

    python main.py cnt --dset hearst_counts.txt.gz

These results can be made machine readable by piping them into `compile_table`:

    python main.py cnt --dset hearst_counts.txt.gz | python compile_table.py

To generate the full table from the report, you may simply use `generate_table.sh`:

    bash generate_table.sh results.json

Please note that due to licensing concerns, we were not able to release our
train/validation/test folds from the paper, so results may differ slightly than
those reported.

## Requirements

The module was developed with python3 in mind, and is not tested for python2.
Nonetheless, cross-platform compatibility may be possible.

The suite requires several packages you probably already have installed:
`numpy`, `scipy`, `pandas`, `scikit-learn` and `nltk`. These can be installed
using pip:

    pip install -r requirements.txt

If you've never used `nltk` before, you'll need to install the wordnet module.

    python -c "import nltk; nltk.download('wordnet')"

## Evaluating your own model

You can evaluate your own model in two separate ways. The simplest way is simply
to create a copy of example.tsv, and fill in your model's predictions in the `sim`
column. You must include a prediction for every pair, but you may set the `is_oov`
column to `1` to ensure it is correctly calculated.

You may then evaluate the model:

    python main.py precomputed --dset example.tsv

You can also implement any model by extending the `base.HypernymySuiteModel` class
and filling in your own implemenation for `predict` or `predict_many`.

## References

If you find this code useful for your research, please cite the following paper:

    @inproceedings{roller2018hearst
        title = {Hearst Patterns Revisited: Automatic Hypernym Detection from Large Text Corpora},
        author = {Roller, Stephen and Kiela, Douwe and Nickel, Maximilian},
        year = {2018},
        booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
        location = {Melbourne, Australia},
        publisher = {Association for Computational Linguistics}
    }

## License

This code is licensed under [CC-BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The data contained in `hearst_counts.txt` was extracted from a combination of
[Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download) and Gigaword.
Please see publication for details.
