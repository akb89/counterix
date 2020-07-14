"""Welcome to counterix.

This is the entry point of the application.
"""

import argparse
import logging
import logging.config

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def main():
    """Launch counterix."""
    parser = argparse.ArgumentParser(prog='counterix')
    subparsers = parser.add_subparsers()
    parser_extract = subparsers.add_parser(
        'extract', formatter_class=argparse.RawTextHelpFormatter,
        help='extract vocab from embeddings txt file')
    parser_extract.set_defaults(func=_extract)

    args = parser.parse_args()
    args.func(args)
