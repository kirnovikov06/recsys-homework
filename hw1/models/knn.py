import numpy as np
from scipy import sparse


class KNN:
    def __init__(self, k=50, shrink=10):
        self.k = k
        self.shrink = shrink

    def fit(self, a):
        self.a = a.tocsr().astype(np.float32)
        g = (self.a.T @ self.a).toarray()
        d = np.sqrt(np.diag(g))
        s = g / (d[:, None] * d[None, :] + self.shrink)
        np.fill_diagonal(s, 0)
        n = s.shape[0]
        k = min(self.k, n - 1)
        j = np.argpartition(-s, k - 1, axis=1)[:, :k]
        i = np.repeat(np.arange(n), k)
        self.w = sparse.csr_matrix((s[i, j.ravel()], (i, j.ravel())), shape=(n, n))
        return self

    def score(self, u):
        return (self.a[u] @ self.w.T).toarray()
