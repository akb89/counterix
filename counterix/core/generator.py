"""A basic count-based model using sparse matrices and no ppmi."""
import logging
from collections import defaultdict

from scipy import sparse
from tqdm import tqdm

import embeddix

logger = logging.getLogger(__name__)

__all__ = ('generate_raw_count_based_dsm')


def count_words(corpus_filepath, min_count):
    """Count words in a corpus."""
    _counts = defaultdict(int)
    logger.info('Counting words in {}'.format(corpus_filepath))
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            for word in line.split():
                _counts[word] += 1
    if min_count == 0:
        counts = _counts
    else:
        counts = {word: count for word, count in _counts.items()
                  if count >= min_count}
        logger.info('Filtering out vocabulary words with counts lower '
                    'than {}, shrinking size by {:.2f}% from {} to {}.'
                    .format(min_count, 100-len(counts)*100.0/len(_counts),
                            len(_counts), len(counts)))
    return counts


def _count_raw_no_filter(corpus_filepath, win_size, word_to_idx_dic,
                         total_num_lines):
    data_dic = defaultdict(lambda: defaultdict(int))
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        for line in tqdm(input_stream, total=total_num_lines):
            tokens = line.strip().split()
            # raw count with symmetric matrix
            for token_pos, token in enumerate(tokens):
                start = 0 if win_size == 0 else max(0, token_pos-win_size)
                while start < token_pos:
                    ctx = tokens[start]
                    if token in word_to_idx_dic and ctx in word_to_idx_dic:
                        token_idx = word_to_idx_dic[token]
                        ctx_idx = word_to_idx_dic[ctx]
                        data_dic[token_idx][ctx_idx] += 1
                    start += 1
        return data_dic


def generate_raw_count_based_dsm(corpus_filepath, min_count, win_size):
    """Generate a count-based distributional model with raw counts."""
    word_to_count_dic = count_words(corpus_filepath=corpus_filepath,
                                    min_count=min_count)
    word_to_idx_dic = {word: idx for idx, word
                       in enumerate(word_to_count_dic.keys())}
    total_num_lines = embeddix.count_lines(corpus_filepath)
    data_dic = _count_raw_no_filter(corpus_filepath, win_size, word_to_idx_dic,
                                    total_num_lines)
    logger.info('Building CSR sparse matrix...')
    rows = []
    columns = []
    data = []
    for row_idx in tqdm(data_dic):
        for col_idx in data_dic[row_idx]:
            rows.append(row_idx)
            columns.append(col_idx)
            data.append(data_dic[row_idx][col_idx])
            # rely on the fact that the matrix is symmetric
            rows.append(col_idx)
            columns.append(row_idx)
            data.append(data_dic[row_idx][col_idx])
    model = sparse.csr_matrix((data, (rows, columns)),
                              shape=(len(word_to_idx_dic),
                                     len(word_to_idx_dic)),
                              dtype='f')
    # logger.info('Matrix info: {} non-zero entries, {} shape, {:.6f} density'
    #             .format(model.getnnz(), model.shape,
    #                     model.getnnz()*1.0/(model.shape[0]*model.shape[1])))
    return model, word_to_idx_dic
