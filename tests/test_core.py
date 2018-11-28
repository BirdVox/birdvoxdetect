import birdvoxdetect
from birdvoxdetect.core import get_output_path, process_file
import pytest
import tempfile
import numpy as np
import os
import pandas as pd
import shutil
import soundfile as sf
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
from birdvoxdetect.birdvoxdetect_warnings import BirdVoxDetectWarning


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
FG_10SEC_PATH = os.path.join(TEST_AUDIO_DIR,
    'BirdVox-scaper_example_foreground.wav')


def test_get_output_path():
    test_filepath = '/path/to/the/test/file/audio.wav'
    suffix = 'timestamps.csv'
    test_output_dir = '/tmp/test/output/dir'
    exp_output_path = '/tmp/test/output/dir/audio_timestamps.csv'
    output_path = get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = '/path/to/the/test/file/audio_timestamps.csv'
    output_path = get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = '/path/to/the/test/file/audio.csv'
    output_path = get_output_path(test_filepath, '.csv')
    assert output_path == exp_output_path


def test_process_file():
    # non-existing path
    invalid_filepath = 'path/to/a/nonexisting/file.wav'
    pytest.raises(BirdVoxDetectError, process_file, invalid_filepath)

    # non-audio path
    nonaudio_existing_filepath = 'README.md'
    pytest.raises(BirdVoxDetectError, process_file, nonaudio_existing_filepath)

    # standard call
    tempdir = tempfile.mkdtemp()
    process_file(FG_10SEC_PATH, output_dir=tempdir)
    csv_path = os.path.join(
        tempdir, 'BirdVox-scaper_example_foreground_timestamps.csv')
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert len(df.columns) == 3
    assert df.columns[1] == "Time (s)"
    assert df.columns[2] == "Likelihood (%)"
    assert df["Time (s)"] == [1.2, 2.6, 3.45]

    # export clips
    tempdir = tempfile.mkdtemp()
    process_file(FG_10SEC_PATH, output_dir=tempdir, export_clips=True)
    clips_dir = os.path.join(
        tempdir, 'BirdVox-scaper_example_foreground_clips')
    assert os.path.exists(clips_dir)
    clips_list = sorted(os.listdir(clips_dir))
    assert len(clips_list) == 3
    assert clips_list[0] == 'BirdVox-scaper_example_foreground_1.20.wav'
    assert clips_list[1] == 'BirdVox-scaper_example_foreground_2.60.wav'
    assert clips_list[2] == 'BirdVox-scaper_example_foreground_3.45.wav'
