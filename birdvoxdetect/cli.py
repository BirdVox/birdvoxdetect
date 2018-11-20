from __future__ import print_function
import os
import sys
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
import birdvoxdetect
from argparse import ArgumentParser, RawDescriptionHelpFormatter,\
    ArgumentTypeError
from collections import Iterable
from six import string_types


def positive_float(value):
    """An argparse-like method for accepting only positive floats"""
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError('Expected a positive float, error message: ' +\
            '{}'.format(e))
    if fvalue <= 0:
        raise ArgumentTypeError('Expected a positive float')
    return fvalue


def get_file_list(input_list):
    """Parse list of input paths."""
    if not isinstance(input_list, Iterable)\
            or isinstance(input_list, string_types):
        raise ArgumentTypeError('input_list must be a non-string iterable')
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
            raise BirdVoxDetectError(
                'Could not find input at path {}'.format(item))

    return file_list


def run(inputs, output_dir=None, suffix=None, hop_size=0.05, verbose=False):
    if isinstance(inputs, string_types):
        file_list = [inputs]
    elif isinstance(inputs, Iterable):
        file_list = get_file_list(inputs)
    else:
        raise BirdVoxDetectError('Invalid input: {}'.format(str(inputs)))

    if len(file_list) == 0:
        print('birdvoxdetect: No WAV files found in {}. Aborting.'.format(
            str(inputs)))
        sys.exit(-1)

    # Process all files in the arguments
    for filepath in file_list:
        if verbose:
            print('birdvoxdetect: Processing: {}'.format(filepath))
        birdvoxdetect.process_file(filepath,
                     output_dir=output_dir,
                     suffix=suffix,
                     hop_size=hop_size,
                     verbose=verbose)
    if verbose:
        print('birdvoxdetect: Done.')


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('inputs', nargs='+',
        help='Path or paths to files to process, or path to '
            'a directory of files to process.')

    parser.add_argument('--output-dir', '-o', default=None,
        help='Directory to save the output file(s); '
            'The default value is the same directory as the input '
            'file(s).')

    parser.add_argument('--export-clips', '-c', action='store_true',
        default=False,
        help='Export detected events as audio clips in WAV format.')

    parser.add_argument('--threshold', '-t', type=positive_float, default=50,
        help='Detection threshold, between 10 and 90. The default value is 50. '
            'Greater values lead to higher precision at the expense '
            'of a lower recall.')

    parser.add_argument('--suffix', '-s', default="",
        help='String to append to the output filenames.'
            'The default value is the empty string.')

    parser.add_argument('--frame-rate', '-r', type=positive_float, default=20.0,
        help='Temporal resolution of the detection curve, '
            'expressed in frames per second (fps). '
            'The default value is 20. We recommend values of 15 or above.')

    parser.add_argument('--clip-duration', '-d', type=positive_float, default=1.0,
        help='Duration of the exported clips, expressed in seconds (fps). '
            'The default value is 1.0, that is, one second. '
            'We recommend values of 0.5 or above.')

    parser.add_argument('--quiet', '-q', action='store_true', default=False,
        help='Suppress all non-error messages to stdout.')
    
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
        help='Print timestamps of detected events.')
    
    parser.add_argument('--version', '-V', action='store_true', default=False,
        help='Print current version.')

    return parser.parse_args(args)


def main():
    """
    Extracts nocturnal flight calls from audio by means of the BirdVoxDetect
    deep learning model (Lostanlen et al. 2019).
    """
    args = parse_args(sys.argv[1:])
    
    if args.version:
        print(birdvoxdetect.version.version)
        

    run(args.inputs,
        output_dir=args.output_dir,
        suffix=args.suffix,
        hop_size=args.hop_size,
        verbose=not args.quiet)
