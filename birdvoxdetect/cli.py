import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError


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


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                        formatter_class=RawDescriptionHelpFormatter)

def main():
    """
    Extracts nocturnal flight calls from audio by means of the BirdVoxDetect deep learning model (Lostanlen et al. 2019).
    """
    args = parse_args(sys.argv[1:])

    raise NotImplementedError()
