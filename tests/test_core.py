import datetime
import h5py
import json
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import soundfile as sf
import tempfile

from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
from birdvoxdetect.core import get_output_path, process_file


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, "data", "audio")

NEGATIVE_MD5 = "ff3d3feb-3371-44ad-a3b3-85969c2cd5ab"
POSITIVE_MD5 = "fd79e55d-d3a3-4083-aba1-4f00b545c3d6"


def test_get_output_path():
    test_filepath = "/path/to/the/test/file/audio.wav"
    suffix = "checklist.csv"
    test_output_dir = "/tmp/test/output/dir"
    exp_output_path = "/tmp/test/output/dir/audio_checklist.csv"
    output_path = get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = "/path/to/the/test/file/audio_checklist.csv"
    output_path = get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = "/path/to/the/test/file/audio.csv"
    output_path = get_output_path(test_filepath, ".csv")
    assert output_path == exp_output_path


def test_process_file():
    # non-existing path
    invalid_filepath = "path/to/a/nonexisting/file.wav"
    pytest.raises(BirdVoxDetectError, process_file, invalid_filepath)

    # non-audio path
    nonaudio_existing_filepath = "/Users/vl238"
    pytest.raises(BirdVoxDetectError, process_file, nonaudio_existing_filepath)

    # non-existing model
    pytest.raises(
        BirdVoxDetectError,
        process_file,
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        detector_name="a_birdvoxdetect_model_that_does_not_exist",
    )

    # non-existing model
    pytest.raises(
        BirdVoxDetectError,
        process_file,
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        detector_name="birdvoxdetect_empty",
    )

    # standard call
    # this example has one flight call (SWTH) at 8.79 seconds
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=os.path.join(tmpdir, "subfolder"),
    )
    csv_path = os.path.join(tmpdir, "subfolder", POSITIVE_MD5 + "_checklist.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert len(df.columns) == 8
    assert df.columns[0] == "Time (hh:mm:ss)"
    assert df.columns[1] == "Detection confidence (%)"
    assert df.columns[2] == "Order"
    assert df.columns[3] == "Order confidence (%)"
    assert df.columns[4] == "Family"
    assert df.columns[5] == "Family confidence (%)"
    assert df.columns[6] == "Species"
    assert df.columns[7] == "Species confidence (%)"

    df_strptime = datetime.datetime.strptime(
        list(df["Time (hh:mm:ss)"])[0], "%H:%M:%S.%f"
    )
    df_timedelta = df_strptime - datetime.datetime.strptime(
        "00:00:00.00", "%H:%M:%S.%f"
    )
    assert np.allclose(
        np.array([df_timedelta.total_seconds()]), np.array([8.79]), atol=0.1
    )
    assert list(df["Order"])[0] == "Passerine"
    assert list(df["Family"])[0] == "Thrush"
    assert list(df["Species"])[0] == "Swainson's thrush"
    shutil.rmtree(tmpdir)

    # standard call on clip without any flight call
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, NEGATIVE_MD5 + ".wav"),
        output_dir=os.path.join(tmpdir, "subfolder"),
    )
    csv_path = os.path.join(tmpdir, "subfolder", NEGATIVE_MD5 + "_checklist.csv")
    df = pd.read_csv(csv_path)
    assert len(df) == 0

    # export clips
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        export_clips=True,
    )
    clips_dir = os.path.join(tmpdir, POSITIVE_MD5 + "_clips")
    assert os.path.exists(clips_dir)
    clips_list = sorted(os.listdir(clips_dir))
    assert len(clips_list) == 1
    assert clips_list[0].startswith(POSITIVE_MD5 + "_00_00_08-78")
    assert clips_list[0].endswith("SWTH.wav")
    assert np.all([c.endswith(".wav") for c in clips_list])
    shutil.rmtree(tmpdir)

    # export confidence
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        export_confidence=True,
    )
    confidence_path = os.path.join(tmpdir, POSITIVE_MD5 + "_confidence.hdf5")
    with h5py.File(confidence_path, "r") as f:
        confidence = f["confidence"][()]
    assert confidence.shape == (199,)
    shutil.rmtree(tmpdir)

    # export context
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        export_context=True,
    )
    context_path = os.path.join(tmpdir, POSITIVE_MD5 + "_context.hdf5")
    assert os.path.exists(context_path)
    with h5py.File(context_path, "r") as f:
        confidence = f["context"][()]
    shutil.rmtree(tmpdir)

    # export list of sensor faults
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        export_faults=True,
    )
    faultlist_path = os.path.join(tmpdir, POSITIVE_MD5 + "_faults.csv")
    assert os.path.exists(faultlist_path)
    faultlist_df = pd.read_csv(faultlist_path)
    columns = faultlist_df.columns
    assert np.all(
        columns
        == np.array(["Start (hh:mm:ss)", "Stop (hh:mm:ss)", "Fault confidence (%)"])
    )
    shutil.rmtree(tmpdir)

    # export probabilities as JSON file
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        predict_proba=True,
    )
    json_path = os.path.join(tmpdir, POSITIVE_MD5 + "_proba.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)
    assert "events" in json_dict.keys()
    assert "metadata" in json_dict.keys()
    assert "taxonomy" in json_dict.keys()
    shutil.rmtree(tmpdir)

    # suffix
    tmpdir = tempfile.mkdtemp()
    process_file(
        os.path.join(TEST_AUDIO_DIR, POSITIVE_MD5 + ".wav"),
        output_dir=tmpdir,
        suffix="mysuffix",
    )
    csv_path = os.path.join(tmpdir, POSITIVE_MD5 + "_mysuffix_checklist.csv")
    assert os.path.exists(csv_path)
    shutil.rmtree(tmpdir)
