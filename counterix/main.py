"""Welcome to counterix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

from scipy import sparse

import counterix.core.generator as generator
import counterix.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _generate(args):
    logger.info('Generating distributional model from {}'.format(args.corpus))
    if args.corpus.endswith('.txt'):
        model_filepath = os.path.abspath(args.corpus)
    else:
        model_filepath = os.path.abspath(args.corpus).split('.txt')
    vocab_filepath = '{}.vocab'.format(model_filepath)
    model, vocab = generator.generate_distributional_model(
        args.corpus, args.min_count, args.win_size)
    logger.info('Saving vocabulary to {}'.format(vocab_filepath))
    with open(vocab_filepath, 'w', encoding='utf-8') as vocab_stream:
        for word, idx in vocab.items():
            print('{}\t{}'.format(idx, word), file=vocab_stream)
    logger.info('Saving numpy matrix to {}'.format(model_filepath))
    sparse.save_npz(model_filepath, model)


def main():
    """Launch counterix."""
    parser = argparse.ArgumentParser(prog='counterix')
    subparsers = parser.add_subparsers()
    parser_generate = subparsers.add_parser(
        'generate', formatter_class=argparse.RawTextHelpFormatter,
        help='generate raw frequency count based model')
    parser_generate.set_defaults(func=_generate)
    parser_generate.add_argument('-c', '--corpus', required=True,
                                 help='an input .txt corpus to compute \
                                 counts on')
    parser_generate.add_argument('-m', '--min-count', default=0, type=int,
                                 help='frequency threshold on vocabulary')
    parser_generate.add_argument('-w', '--win-size', default=2, type=int,
                                 help='size of context window')
    args = parser.parse_args()
    args.func(args)
