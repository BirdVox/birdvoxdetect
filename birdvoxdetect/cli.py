from __future__ import print_function
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from collections.abc import Iterable
import logging
import numpy as np
import os
from six import string_types
import sys

import birdvoxdetect
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError

# The following line circumvent issue #1715 in xgboost
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_file_list(input_list):
    """Parse list of input paths."""
    if not isinstance(input_list, Iterable) or isinstance(input_list, string_types):
        raise ArgumentTypeError("input_list must be a non-string iterable")
    file_list = []
    for item in input_list:
        if os.path.isfile(item):
            file_list.append(os.path.abspath(item))
        elif os.path.isdir(item):
            for fname in os.listdir(item):
                path = os.path.join(item, fname)
                if os.path.isfile(path):
                    file_list.append(path)
        else:
            raise BirdVoxDetectError("Could not find input at path {}".format(item))

    return file_list


def run(
    inputs,
    output_dir=None,
    export_clips=False,
    export_confidence=False,
    export_faults=False,
    export_logger=False,
    predict_proba=False,
    threshold=50.0,
    suffix="",
    clip_duration=1.0,
    logger_level=logging.INFO,
):
    verbose = True
    if isinstance(inputs, string_types):
        file_list = [inputs]
    elif isinstance(inputs, Iterable):
        file_list = get_file_list(inputs)
    else:
        raise BirdVoxDetectError("Invalid input: {}".format(str(inputs)))

    if len(file_list) == 0:
        print("birdvoxdetect: No WAV files found in {}. Aborting.".format(str(inputs)))
        sys.exit(-1)

    # Print header
    if verbose:
        if threshold:
            print("birdvoxdetect: Threshold = {:4.1f}".format(threshold))

        if output_dir:
            print("birdvoxdetect: Output directory = " + output_dir)

        if not suffix == "":
            print("birdvoxdetect: Suffix string = " + suffix)

        if export_clips:
            export_clips_str = "".join(
                [
                    "Duration of exported clips = ",
                    "{:.2f} seconds.".format(clip_duration),
                ]
            )
            print("birdvoxdetect: " + export_clips_str)

    # Process all files in the arguments
    for filepath in file_list:
        if verbose:
            print("birdvoxdetect: Processing: {}".format(filepath))
        birdvoxdetect.process_file(
            filepath,
            clip_duration=clip_duration,
            export_clips=export_clips,
            export_confidence=export_confidence,
            export_faults=export_faults,
            export_logger=export_logger,
            logger_level=logger_level,
            output_dir=output_dir,
            predict_proba=predict_proba,
            suffix=suffix,
            threshold=threshold,
        )
    if verbose:
        print("birdvoxdetect: Done.")


def parse_args(args):
    parser = ArgumentParser(
        sys.argv[0],
        description=main.__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help="Path or paths to files to process, or path to "
        "a directory of files to process.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save the output file(s); "
        "The default value is the same directory as the input "
        "file(s).",
    )

    parser.add_argument(
        "--export-clips",
        "-c",
        action="store_true",
        help="Export detected events as audio clips in WAV format.",
    )

    parser.add_argument(
        "--export-confidence",
        "-C",
        action="store_true",
        help="Export the time series of model confidence values of events"
        "in HDF5 format.",
    )

    parser.add_argument(
        "--export-faults",
        "-f",
        action="store_true",
        help="Export list of sensor faults in CSV format.",
    )

    parser.add_argument(
        "--export-logger",
        "-l",
        action="store_true",
        help="Export output of Python logger in TXT format.",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=valid_threshold,
        default=50,
        help="Detection threshold, between 10 and 90. "
        "The default value is 50. "
        "Greater values lead to higher precision at the expense "
        "of a lower recall.",
    )

    parser.add_argument(
        "--suffix",
        "-s",
        default="",
        help="String to append to the output filenames."
        "The default value is the empty string.",
    )

    parser.add_argument(
        "--clip-duration",
        "-d",
        type=positive_float,
        default=None,
        help="Duration of the exported clips, expressed in seconds (fps). "
        "The default value is 1.0, that is, one second. "
        "We recommend values of 0.5 or above.",
    )

    parser.add_argument(
        "--predict-proba",
        "-p",
        action="store_true",
        help="Export output probabilities as a JSON container.",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Print less messages on screen."
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print timestamps of detected events.",
    )

    parser.add_argument(
        "--version", "-V", action="store_true", help="Print version number."
    )

    if args == []:
        parser.print_help(sys.stdout)
        return ""

    args = parser.parse_args(args)

    if args.quiet and args.verbose:
        raise BirdVoxDetectError(
            "Command-line flags --quiet (-q) and --verbose (-v) "
            "are mutually exclusive."
        )

    if args.clip_duration is None:
        args.clip_duration = 1.0
    elif not args.export_clips:
        raise BirdVoxDetectError(
            "The --export-clips (-c) flag should be present "
            "if the --clip-duration (-d) flag is present."
        )

    return args


def main():
    """
    Extracts nocturnal flight calls from audio by means of the BirdVoxDetect
    deep learning model (Lostanlen et al. 2019).
    """
    args = parse_args(sys.argv[1:])

    if args == "":
        return

    if args.version:
        print(birdvoxdetect.version.version)
        return

    if args.quiet:
        logger_level = 30
    elif args.verbose:
        logger_level = 10
    else:
        logger_level = 25

    run(
        args.inputs,
        output_dir=args.output_dir,
        export_clips=args.export_clips,
        export_confidence=args.export_confidence,
        export_faults=args.export_faults,
        export_logger=args.export_logger,
        predict_proba=args.predict_proba,
        threshold=args.threshold,
        suffix=args.suffix,
        clip_duration=args.clip_duration,
        logger_level=logger_level,
    )


def positive_float(value):
    """An argparse-like method for accepting only positive number"""
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError(
            "Expected a positive float, error message: {}".format(e)
        )
    if np.isnan(fvalue) or fvalue <= 0:
        raise ArgumentTypeError("Expected a positive number")
    return fvalue


def valid_threshold(value):
    """An argparse-like method for accepting only floats between 0 and 100"""
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError(
            "Expected a positive float, error message: {}".format(e)
        )
    if np.isnan(fvalue) or fvalue < 0 or fvalue > 100:
        raise ArgumentTypeError("Expected a number between 0 and 100")
    return fvalue
