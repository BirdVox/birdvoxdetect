import openl3
import pytest
import tempfile
import numpy as np
import os
import shutil
import soundfile as sf
from openl3.openl3_exceptions import OpenL3Error
from openl3.openl3_warnings import OpenL3Warning


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')


def test_get_output_path():
    test_filepath = '/path/to/the/test/file/audio.wav'
    suffix = 'timestamps.csv'
    test_output_dir = '/tmp/test/output/dir'
    exp_output_path = '/tmp/test/output/dir/audio_timestamps.csv'
    output_path = openl3.get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = '/path/to/the/test/file/audio_timestamps.csv'
    output_path = openl3.get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = '/path/to/the/test/file/audio.csv'
    output_path = openl3.get_output_path(test_filepath, '.csv')
    assert output_path == exp_output_path
