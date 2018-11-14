from __future__ import print_function
import os
import sys
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError
import birdvoxdetect
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from collections import Iterable
from six import string_types


def positive_float(value):
    """An argparse-like method for accepting only positive floats"""
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError('Expected a positive float, error message: '
                                '{}'.format(e))
    if fvalue <= 0:
        raise ArgumentTypeError('Expected a positive float')
    return fvalue


def get_file_list(input_list):
    """Parse list of input paths."""
    if not isinstance(input_list, Iterable) or isinstance(input_list, string_types):
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
            raise BirdVoxDetectError('Could not find input at path {}'.format(item))

    return file_list


def run(inputs, output_dir=None, suffix=None, hop_size=0.05, verbose=False):
    if isinstance(inputs, string_types):
        file_list = [inputs]
    elif isinstance(inputs, Iterable):
        file_list = get_file_list(inputs)
    else:
        raise BirdVoxDetectError('Invalid input: {}'.format(str(inputs)))

    if len(file_list) == 0:
        print('birdvoxdetect: No WAV files found in {}. Aborting.'.format(str(inputs)))
        sys.exit(-1)
    raise NotImplementedError()


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('inputs', nargs='+',
                        help='Path or paths to files to process, or path to '
                             'a directory of files to process.')

    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory to save the ouptut file(s); '
                             'if not given, the output will be '
                             'saved to the same directory as the input WAV '
                             'file(s).')

    parser.add_argument('--suffix', '-x', default=None,
                        help='String to append to the output filenames.'
                             'If not provided, no suffix is added.')

    parser.add_argument('--hop-size', '-t', type=positive_float, default=0.05,
                    help=['Hop size in seconds for processing audio files. ',
                    'We recommend values of 0.075 or smaller'])

    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                    help='Suppress all non-error messages to stdout.')

    return parser.parse_args(args)

def main():
    """
    Extracts nocturnal flight calls from audio by means of the BirdVoxDetect deep learning model (Lostanlen et al. 2019).
    """
    args = parse_args(sys.argv[1:])

    run(args.inputs,
        output_dir=args.output_dir,
        suffix=args.suffix,
        hop_size=args.hop_size,
        verbose=not args.quiet)
