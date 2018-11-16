import pytest
import os
from birdvoxdetect.cli import positive_float, get_file_list, run
from argparse import ArgumentTypeError
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
import tempfile
import numpy as np
import shutil
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
NOISY_1MIN_24K_PATH = os.path.join(TEST_AUDIO_DIR,
    'BirdVox-full-night_unit03_00-19-45_01min.wav')
CLEAN_1MIN_PATH = os.path.join(TEST_AUDIO_DIR,
    'CLO-43SD_synth-clean_01min.wav')


def test_positive_float():

    # test that returned value is float
    f = positive_float(5)
    assert f == 5.0
    assert type(f) is float

    # test it works for valid strings
    f = positive_float('1.3')
    assert f == 1.3
    assert type(f) is float

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, 'hello']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_float, i)



def test_get_file_list():

    # test for invalid input (must be iterable, e.g. list)
    pytest.raises(ArgumentTypeError, get_file_list,
        NOISY_1MIN_24K_PATH)

    # test for valid list of file paths
    flist = get_file_list(
        [CLEAN_1MIN_PATH, NOISY_1MIN_24K_PATH])
    assert len(flist) == 2
    assert flist[0] == CLEAN_1MIN_PATH
    assert flist[1] == NOISY_1MIN_24K_PATH


def test_run(capsys):

    # test invalid input
    invalid_inputs = [None, 5, 1.0]
    for i in invalid_inputs:
        pytest.raises(BirdVoxDetectError, run, i)
