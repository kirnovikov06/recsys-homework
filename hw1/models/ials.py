import numpy as np
from scipy import sparse
from scipy.linalg import solve


class IALS:
    def __init__(self, k=32, reg=10, alpha=20, epochs=12, seed=42):
        self.k = k
        self.reg = reg
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed

    def target(self, r):
        return (r >= 4).astype(np.float64)

    def step(self, a, y):
        x = np.zeros((a.shape[0], self.k))
        g = y.T @ y + self.reg * np.eye(self.k)
        for u in range(a.shape[0]):
            l, r = a.indptr[u:u + 2]
            j = a.indices[l:r]
            if not len(j):
                continue
            z = y[j]
            b = (1 + self.alpha) * z.T @ self.target(a.data[l:r])
            h = g + self.alpha * (z.T @ z)
            x[u] = solve(h, b, assume_a="pos", check_finite=False)
        return x

    def fit(self, a):
        a = sparse.csr_matrix(a, dtype=np.float64)
        b = a.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.y = rng.normal(0, 0.03, (a.shape[1], self.k))
        for _ in range(self.epochs):
            self.x = self.step(a, self.y)
            self.y = self.step(b, self.x)
        return self

    def score(self, u):
        return self.x[u] @ self.y.T
