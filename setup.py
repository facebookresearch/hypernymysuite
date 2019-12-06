#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from setuptools import setup, find_packages, Extension
import sys


with open('README.md') as f:
    readme = f.read()


setup(
    name='hypernymysuite',
    version='0.0.1',
    description='Hearst Patterns Revisited: Automatic Hypernym Detection from Large Text Corpora',
    url='https://github.com/facebookresearch/hypernymysuite.git',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'nltk',
        'pandas',
    ],
    packages=find_packages(),
)
