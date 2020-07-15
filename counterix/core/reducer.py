"""Apply SVD."""
import logging
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

__all__ = ('apply_sparse_svd')


# pylint: disable=C0103
def apply_sparse_svd(M, dim):
    """Apply SVD to sparse CSR matrix."""
    if dim == 0 or dim >= min(M.shape):
        logger.warning('Specified k={} null or exceeds matrix shape limit = '
                       '{}. Resetting k to {}'.format(dim, min(M.shape),
                                                      min(M.shape) - 1))
        dim = min(M.shape) - 1
    logger.info('Running SVD...')
    U, S, _ = svds(M, k=dim, which='LM', return_singular_vectors='u')
    S = S[::-1]  # put singular values in decreasing order of values
    U = U[:, ::-1]  # put singular vectors in decreasing order of sing. values
    return U, S
