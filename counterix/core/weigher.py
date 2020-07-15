"""PPMI weighing func."""

import logging
import numpy as np
import scipy.sparse as sparse

logger = logging.getLogger(__name__)

__all__ = ('weigh')


def weigh(csr_matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logger.info('Weighing raw count CSR matrix via PPMI')
    words = sparse.csr_matrix(csr_matrix.sum(axis=1))
    contexts = sparse.csr_matrix(csr_matrix.sum(axis=0))
    total_sum = csr_matrix.sum()
    # csr_matrix = csr_matrix.multiply(words.power(-1)) # #(w, c) / #w
    # csr_matrix = csr_matrix.multiply(contexts.power(-1))  # #(w, c) / (#w * #c)
    # csr_matrix = csr_matrix.multiply(total)  # #(w, c) * D / (#w * #c)
    csr_matrix = csr_matrix.multiply(words.power(-1))\
                           .multiply(contexts.power(-1))\
                           .multiply(total_sum)
    csr_matrix.data = np.log2(csr_matrix.data)  # PMI = log(#(w, c) * D / (#w * #c))
    csr_matrix = csr_matrix.multiply(csr_matrix > 0)  # PPMI
    csr_matrix.eliminate_zeros()
    return csr_matrix
