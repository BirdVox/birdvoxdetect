from argparse import ArgumentTypeError
import os
import pytest
import shutil
import tempfile


try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch


from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
from birdvoxdetect.cli import get_file_list, main
from birdvoxdetect.cli import parse_args, positive_float, run, valid_threshold


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
NEGATIVE_PATH = os.path.join(TEST_AUDIO_DIR,
    'ff3d3feb-3371-44ad-a3b3-85969c2cd5ab.wav')
POSITIVE_PATH = os.path.join(TEST_AUDIO_DIR,
    'fd79e55d-d3a3-4083-aba1-4f00b545c3d6.wav')

def test_get_file_list():

    # test for invalid input (must be iterable, e.g. list)
    pytest.raises(ArgumentTypeError, get_file_list,
        POSITIVE_PATH)

    # test for valid list of file paths
    flist = get_file_list([NEGATIVE_PATH, POSITIVE_PATH])
    assert len(flist) == 2
    assert flist[0] == NEGATIVE_PATH
    assert flist[1] == POSITIVE_PATH

    # test for valid folder
    flist = get_file_list([TEST_AUDIO_DIR])
    assert len(flist) == 2
    flist = sorted(flist)
    assert flist[0] == POSITIVE_PATH
    assert flist[1] == NEGATIVE_PATH

    # combine list of files and folders
    flist = get_file_list([TEST_AUDIO_DIR, POSITIVE_PATH])
    assert len(flist) == 3

    # nonexistent path
    pytest.raises(BirdVoxDetectError, get_file_list, ['/fake/path/to/file'])


def test_parse_args():

    # test default values
    args = [POSITIVE_PATH]
    args = parse_args(args)
    assert args.output_dir is None
    assert args.export_clips == False
    assert args.export_confidence == False
    assert args.export_faults == False
    assert args.export_logger == False
    assert args.threshold == 50.0
    assert args.suffix == ""
    assert args.clip_duration == 1.0
    assert args.quiet is False
    assert args.verbose is False

    # test custom values
    args = [POSITIVE_PATH,
            '-o', '/output/dir',
            '-c',
            '-C',
            '-f',
            '-l',
            '-t', '60',
            '-s', 'mysuffix',
            '-d', '0.5',
            '-q']
    args = parse_args(args)
    assert args.output_dir == '/output/dir'
    assert args.export_clips == True
    assert args.export_confidence == True
    assert args.export_faults == True
    assert args.export_logger == True
    assert args.threshold == 60.0
    assert args.suffix == 'mysuffix'
    assert args.clip_duration == 0.5
    assert args.quiet is True

    # test clash between quiet and verbose
    args = [POSITIVE_PATH,
           '-v',
           '-q']
    pytest.raises(BirdVoxDetectError, parse_args, args)

    # test clash between absence of export_clips
    # and presence of clip duration
    args = [POSITIVE_PATH,
            '-d', '0.5']
    pytest.raises(BirdVoxDetectError, parse_args, args)


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
    invalid = [-5, -1.0, None, 'nan']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_float, i)


def test_valid_threshold():

    # test that returned value is float
    f = valid_threshold(60)
    assert f == 60.0
    assert type(f) is float

    # test it works for valid strings
    f = valid_threshold('70.3')
    assert f == 70.3
    assert type(f) is float

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, -0.01, None, 100.01, 'nan']
    for i in invalid:
        pytest.raises(ArgumentTypeError, valid_threshold, i)


def test_run(capsys):
    # test invalid input
    invalid_inputs = [None, 5, 1.0]
    for i in invalid_inputs:
        pytest.raises(BirdVoxDetectError, run, i)

    # test empty input folder
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        tempdir = tempfile.mkdtemp()
        run([tempdir])
    shutil.rmtree(tempdir)

    # make sure it exited
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == -1

    # make sure it printed a message
    captured = capsys.readouterr()
    expected_message =\
        'birdvoxdetect: No WAV files found in {}. Aborting.\n'.format(
        str([tempdir]))
    assert captured.out == expected_message

    # test string input
    string_input = POSITIVE_PATH
    tempdir = tempfile.mkdtemp()
    run(string_input, output_dir=tempdir)
    csv_path = os.path.join(
        tempdir, 'fd79e55d-d3a3-4083-aba1-4f00b545c3d6_checklist.csv')
    assert os.path.exists(csv_path)
    shutil.rmtree(tempdir)



def test_script_main(capsys):
    # Duplicate regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch(
            'sys.argv',
            ['birdvoxdetect', POSITIVE_PATH, '--output-dir', tempdir]):
        import birdvoxdetect.__main__

    # Check output file created
    outfile = os.path.join(
        tempdir, 'fd79e55d-d3a3-4083-aba1-4f00b545c3d6_checklist.csv')
    assert os.path.isfile(outfile)
    shutil.rmtree(tempdir)
