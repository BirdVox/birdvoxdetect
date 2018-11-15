# BirdVoxDetect: robust sound event detection of bird flight calls
 [![Documentation Status](https://readthedocs.org/projects/birdvoxdetect/badge/?version=latest)](http://birdvoxdetect.readthedocs.io/en/latest/?badge=latest)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.org/BirdVox/birdvoxdetect.svg?branch=master)](https://travis-ci.org/BirdVox/birdvoxdetect)
[![Coverage Status](https://coveralls.io/repos/github/BirdVox/birdvoxdetect/badge.svg?branch=master)](https://coveralls.io/repos/github/BirdVox/birdvoxdetect?branch=master)
[![PyPI](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg)]()


For details about the deep learning model in BirdVoxDetect and how it was trained, we refer the reader to:

Robust audio detection in bioacoustic sensor networks.<br/>
Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, Steve Kelling, and Juan Pablo Bello<br/>
Under review, 2018.


# Installation instructions

Dependencies
------------
#### TensorFlow
Because TensorFlow comes in CPU-only and GPU-enabled variants, we leave it up to the user to install the version that best fits
their use case.

On most platforms, either of the following commands should properly install TensorFlow:

    pip install tensorflow # CPU-only version
    pip install tensorflow-gpu # GPU-enabled version

For more detailed information, please consult the
[installation instructions of TensorFlow](https://www.tensorflow.org/install/).

#### libsndfile (Linux only)
BirdVoxDetect depends on the PySoundFile module to load audio files, which itself depends on the non-Python library libsndfile.
On Windows and Mac OS X, these will be installed automatically via the ``pip`` package manager and you can therefore skip this step.
However, on Linux, `libsndfile` must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

    apt-get install libsndfile

For more detailed information, please consult the
[installation instructions of pysoundfile](https://pysoundfile.readthedocs.io/en/0.9.0/#installation>).


Installing BirdVoxDetect
------------------------
The simplest way to install BirdVoxDetect is by using ``pip``, which will also install the additional required dependencies
if needed. To install the latest stable version of BirdVoxDetect using ``pip``, simply run

    pip install birdvoxdetect

To install the latest version of BirdVoxDetect from source:

1. Clone or pull the latest version:

        git clone git@github.com:BirdVox/birdvoxdetect.git

2. Install using pip to handle Python dependencies:

        cd birdvoxdetect
        pip install -e .
