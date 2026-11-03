import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, a):
        self.a = a.tocsr().astype(np.float64)
        g = (self.a.T @ self.a).toarray()
        g.flat[::g.shape[0] + 1] += self.reg
        p = cho_solve(cho_factor(g), np.eye(g.shape[0]))
        self.w = -p / np.diag(p)[None, :]
        np.fill_diagonal(self.w, 0)
        return self

    def score(self, u):
        return np.asarray(self.a[u] @ self.w)
