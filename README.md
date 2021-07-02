# BirdVoxDetect: detection and classification of flight calls

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)

BirdVoxDetect is a pre-trained deep learning system which detects flight calls from songbirds in audio recordings, and retrieves the corresponding species.
It relies on per-channel energy normalization (PCEN) and context-adaptive convolutional neural networks (CA-CNN) for improved robustness to background noise.
It is made available both as a Python library and as a command-line tool for Windows, OS X, and GNU/Linux.


## Installation

The simplest way to install BirdVoxDetect is by using the ``pip`` package management system, which will also install the additional required dependencies
if needed.

    pip install birdvoxdetect

 Note that birdvoxdetect requires:
* Python (3.6, 3.7, or 3.8)
* librosa (==0.7.0)
* tensorflow (>=2.2)
* scikit-learn (==0.21.2)
* birdvoxclassify (>=0.3)
* h5py
* pandas


## Usage

### From the command line

To analyze one file:

    birdvoxdetect path/to/file.wav

To analyze multiple files:

    birdvoxdetect path/to/file1.wav path/to/file2.wav

To analyze one folder:

    birdvoxdetect path/to/folder

On Windows:

    birdvoxdetect path\to\folder

Optional arguments:

    --clip-duration CLIP_DURATION, -d CLIP_DURATION
                          Duration of the exported clips, expressed in seconds
                          (fps). The default value is 1.0, that is, one second.
                          We recommend values of 0.5 or above.
    --export-clips, -c    Export detected events as audio clips in WAV format.
    --export-confidence, -C
                          Export the time series of model confidence values of
                          events in HDF5 format.
    --export-faults, -f   Export list of sensor faults in CSV format.
    --export-logger, -l   Export output of Python logger in TXT format.
    --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                          Directory to save the output file(s); The default
                          value is the same directory as the input file(s).
    --predict-proba, -p   Export output probabilities in JSON format.
    --quiet, -q           Print less messages on screen.
    --suffix SUFFIX, -s SUFFIX
                          String to append to the output filenames.The default
                          value is the empty string.
    --threshold THRESHOLD, -t THRESHOLD
                          Detection threshold, between 10 and 90. The default
                          value is 50. Greater values lead to higher precision
                          at the expense of a lower recall.
    --verbose, -v         Print timestamps of detected events.
    --version, -V         Print version number.


### From Python

Call syntax:

    import birdvoxdetect as bvd    
    df = bvd.process_file('path/to/file.wav')

`df` is a Pandas DataFrame with three columns: time, detection confidence, and species.

Below is a typical output from the test suite (file path `tests/data/audio/fd79e55d-d3a3-4083-aba1-4f00b545c3d6.wav`):

    Time (hh:mm:ss),Detection confidence (%),Order,Order confidence (%),Family,Family confidence (%),Species (English name),Species (scientific name),Species (4-letter code),Species confidence (%)
    0,00:00:08.78,70.15%,Passeriformes,100.00%,Turdidae,100.00%,Swainson's thrush,Catharus ustulatus,SWTH,99.28%


## Contact

### Official website
Please visit our website for more information on the BirdVox project: [https://wp.nyu.edu/birdvox](https://wp.nyu.edu/birdvox)

The main developer of BirdVoxDetect is Vincent Lostanlen, scientist at CNRS, the French national center for scientific research.

### Discussion group

For any questions or announcements related to BirdVoxDetect, please refer to our discussion group:
[https://groups.google.com/g/birdvox](https://groups.google.com/g/birdvox)

### References
Please cite the following paper when using BirdVoxDetect in your work:

**[Robust Sound Event Detection in Bioacoustic Sensor Networks](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0214168&type=printable)**<br/>
Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, Steve Kelling, and Juan Pablo Bello<br/>
PLoS ONE 14(10): e0214168, 2019. DOI: https://doi.org/10.1371/journal.pone.0214168


As of v0.4, species classification in BirdVoxDetect relies on a taxonomical neural network (TaxoNet), which is distributed as part of the BirdVoxClassify package. For more details on TaxoNet, please refer to:

**[Chirping up the Right Tree: Incorporating Biological Taxonomies into Deep Bioacoustic Classifiers](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_taxonet_icassp_2020.pdf)**<br/>
Jason Cramer, Vincent Lostanlen, Andrew Farnsworth, Justin Salamon, and Juan Pablo Bello<br/>
In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, May 2020.
