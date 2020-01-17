# BirdVoxDetect: detection and classification of flight calls

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.org/BirdVox/birdvoxdetect.svg?branch=master)](https://travis-ci.org/BirdVox/birdvoxdetect)

BirdVoxDetect is a pre-trained deep learning system which detects flight calls from songbirds in audio recordings, and retrieves the corresponding species.
It relies on per-channel energy normalization (PCEN) and context-adaptive convolutional neural networks (CA-CNN) for improved robustness to background noise.
It is made available both as a Python library and as a command-line tool for Windows, OS X, and Linux.


## Installation

The simplest way to install BirdVoxDetect is by using the ``pip`` package management system, which will also install the additional required dependencies
if needed.

    pip install birdvoxdetect

 Note that birdvoxdetect requires:
* Python (==3.6)
* birdvoxclassify
* h5py (>=2.9)
* librosa (==0.7.0)
* numpy (==1.16.4)
* pandas (==0.25.1)
* scikit-learn (==0.21.2)
* tensorflow (==1.15)


## Usage

### From the command line

To analyze one file:

    python -m birdvoxdetect /path/to/file.wav

To analyze multiple files:

    python -m birdvoxdetect /path/to/file1.wav /path/to/file2.wav

To analyze one folder:

   python -m birdvoxdetect /path/to/folder

Optional arguments:

    --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                          Directory to save the output file(s); The default
                          value is the same directory as the input file(s).
    --export-clips, -c    Export detected events as audio clips in WAV format.
    --export-confidence, -C
                          Export the time series of model confidence values of
                          eventsin HDF5 format.
    --threshold THRESHOLD, -t THRESHOLD
                          Detection threshold, between 10 and 90. The default
                          value is 30. Greater values lead to higher precision
                          at the expense of a lower recall.
    --suffix SUFFIX, -s SUFFIX
                          String to append to the output filenames.The default
                          value is the empty string.
    --clip-duration CLIP_DURATION, -d CLIP_DURATION
                          Duration of the exported clips, expressed in seconds
                          (fps). The default value is 1.0, that is, one second.
                          We recommend values of 0.5 or above.
    --quiet, -q           Print less messages on screen.
    --verbose, -v         Print timestamps of detected events.
    --version, -V         Print version number.


### From Python

Call syntax:

    import birdvoxdetect as bvd    
    df = bvd.process_file('path/to/file.wav')

`df` is a Pandas DataFrame with three columns: time, detection confidence, and species.

Below is a typical output from the test suite (file `fd79e55d-d3a3-4083-aba1-4f00b545c3d6.wav`):

       Time (hh:mm:ss) Species (4-letter code)  Confidence (%)
    0     00:00:08.78                    SWTH           100.0


## Contact

Vincent Lostanlen, Cornell Lab of Ornithology (`@lostanlen` on GitHub).
For more information on the BirdVox project, please visit our website: [https://wp.nyu.edu/birdvox](https://wp.nyu.edu/birdvox)

Please cite the following paper when using BirdVoxDetect in your work:


**[Robust Sound Event Detection in Bioacoustic Sensor Networks](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0214168&type=printable)**<br/>
Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, Steve Kelling, and Juan Pablo Bello<br/>
PLoS ONE 14(10): e0214168, 2019. DOI: https://doi.org/10.1371/journal.pone.0214168
