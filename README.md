# BirdVoxDetect: detection and classification of flight calls

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.org/BirdVox/birdvoxdetect.svg?branch=master)](https://travis-ci.org/BirdVox/birdvoxdetect)

BirdVoxDetect is a pre-trained deep learning system which detects flight calls from songbirds in audio recordings, and retrieves the corresponding species.
It relies on per-channel energy normalization (PCEN) and context-adaptive convolutional neural networks (CA-CNN) for improved robustness to background noise.
It is made available both as a Python library and as a command-line tool for Windows, OS X, and Linux.




# Installation instructions

The simplest way to install BirdVoxDetect is by using the ``pip`` package management system, which will also install the additional required dependencies
if needed.

    pip install birdvoxdetect



# Acknowledging BirdVoxDetect

Please cite the following paper when using BirdVoxDetect in your work:


**[Robust Sound Event Detection in Bioacoustic Sensor Networks](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0214168&type=printable)**<br/>
Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, Steve Kelling, and Juan Pablo Bello<br/>
PLoS ONE 14(10): e0214168, 2019. DOI: https://doi.org/10.1371/journal.pone.0214168
