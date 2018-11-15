import pytest
import os
from birdvoxdetect.cli import positive_float
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
