"""Welcome to counterix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import embeddix

import counterix.core.generator as generator
import counterix.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _generate(corpus_filepath, min_count, win_size):
    logger.info('Generating distributional model from {}'
                .format(corpus_filepath))
    if not corpus_filepath.endswith('.txt'):
        model_filepath = os.path.abspath(corpus_filepath)
    else:
        model_filepath = os.path.abspath(corpus_filepath).split('.txt')[0]
    model_filepath = '{}.mc-{}.win-{}'.format(model_filepath, min_count,
                                              win_size)
    vocab_filepath = '{}.vocab'.format(model_filepath)
    model, vocab = generator.generate_raw_count_based_dsm(
        corpus_filepath, min_count, win_size)
    embeddix.save_vocab(vocab_filepath, vocab)
    embeddix.save_sparse(model_filepath, model)


def generate(args):
    """Generate raw count model in scipy sparse CSR matrix format."""
    return _generate(args.corpus, args.min_count, args.win_size)


def main():
    """Launch counterix."""
    parser = argparse.ArgumentParser(prog='counterix')
    subparsers = parser.add_subparsers()
    parser_generate = subparsers.add_parser(
        'generate', formatter_class=argparse.RawTextHelpFormatter,
        help='generate raw frequency count based model')
    parser_generate.set_defaults(func=generate)
    parser_generate.add_argument('-c', '--corpus', required=True,
                                 help='absolute filepath to corpus .txt file')
    parser_generate.add_argument('-m', '--min-count', default=0, type=int,
                                 help='frequency threshold on vocabulary')
    parser_generate.add_argument('-w', '--win-size', default=2, type=int,
                                 help='size of context window')
    args = parser.parse_args()
    args.func(args)
