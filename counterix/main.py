"""Welcome to counterix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import embeddix

import counterix.core.generator as generator
import counterix.core.weigher as weigher
import counterix.core.reducer as reducer
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
    logger.info('Saving vocabulary to file...')
    embeddix.save_vocab(vocab_filepath, vocab)
    logger.info('Saving raw count sparse matrix to file...')
    embeddix.save_sparse(model_filepath, model)


def generate(args):
    """Generate raw count model in scipy sparse CSR matrix format."""
    _generate(args.corpus, args.min_count, args.win_size)


def _weigh(model_filepath):
    logger.info('Applying PPMI-weighing to model {}'.format(model_filepath))
    model = embeddix.load_sparse(model_filepath)
    output_filepath = '{}.ppmi'.format(
        os.path.abspath(model_filepath).split('.npz')[0])
    ppmi_weighed_matrix = weigher.weigh(model)
    logger.info('Saving PPMI sparse matrix to file...')
    embeddix.save_sparse(output_filepath, ppmi_weighed_matrix)


def weigh(args):
    """Apply PPMI-weighing to input model."""
    _weigh(args.model)


# pylint: disable=C0103
def _svd(model_filepath, k):
    logger.info('Applying SVD with k={} to model {}'.format(k, model_filepath))
    sing_vectors_filepath = '{}.k-{}.singvectors'.format(
        os.path.abspath(model_filepath).split('.npz')[0], k)
    sing_values_filepath = '{}.k-{}.singvalues'.format(
        os.path.abspath(model_filepath).split('.npz')[0], k)
    model = embeddix.load_sparse(model_filepath)
    sing_vectors, sing_values = reducer.apply_sparse_svd(model, k)
    logger.info('Saving singular vectors to file...')
    embeddix.save_dense(sing_vectors_filepath, sing_vectors)
    logger.info('Saving singular values to file...')
    embeddix.save_dense(sing_values_filepath, sing_values)


def svd(args):
    """Apply SVD."""
    _svd(args.model, args.dim)


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
    parser_weigh = subparsers.add_parser(
        'weigh', formatter_class=argparse.RawTextHelpFormatter,
        help='apply PPMI weighing to input sparse CSR matrix')
    parser_weigh.set_defaults(func=weigh)
    parser_weigh.add_argument('-m', '--model', required=True,
                              help='absolute path to .npz matrix '
                              'corresponding to the distributional '
                              'space to weigh')
    parser_svd = subparsers.add_parser(
        'svd', formatter_class=argparse.RawTextHelpFormatter,
        help='apply svd to input matrix')
    parser_svd.set_defaults(func=svd)
    parser_svd.add_argument('-m', '--model', required=True,
                            help='absolute path to .npz matrix '
                                 'corresponding to the distributional '
                                 'space to reduce via svd')
    parser_svd.add_argument('-k', '--dim', default=0, type=int,
                            help='number of dimensions in final model')
    args = parser.parse_args()
    args.func(args)
