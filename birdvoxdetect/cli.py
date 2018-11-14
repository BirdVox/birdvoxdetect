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


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                        formatter_class=RawDescriptionHelpFormatter)

def main():
    """
    Extracts nocturnal flight calls from audio by means of the BirdVoxDetect deep learning model (Lostanlen et al. 2019).
    """
    args = parse_args(sys.argv[1:])

    raise NotImplementedError()
