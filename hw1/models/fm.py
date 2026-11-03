import numpy as np
import torch
from torch import nn


class FM(nn.Module):
    def __init__(self, k=32, lr=0.01, reg=1e-5, epochs=10, seed=42, meta=True):
        super().__init__()
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.meta = meta

    def forward(self, u, i):
        f = torch.cat((u[:, None] + 1, self.f[i]), dim=1)
        v = self.v(f)
        s = v.sum(dim=1)
        return self.b + self.w(f).sum(dim=(1, 2)) + 0.5 * (s.square() - v.square().sum(dim=1)).sum(dim=1)

    def fit(self, a, cats):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, m = a.shape
        self.n = n
        c = cats if self.meta else [[] for _ in range(m)]
        d = max((max(x, default=-1) for x in c), default=-1) + 1
        f = np.zeros((m, 1 + max(map(len, c), default=0)), dtype=np.int64)
        f[:, 0] = n + np.arange(m) + 1
        for i, v in enumerate(c):
            f[i, 1:len(v) + 1] = n + m + np.asarray(v) + 1
        self.register_buffer("f", torch.from_numpy(f))
        self.v = nn.Embedding(n + m + d + 1, self.k, padding_idx=0)
        self.w = nn.Embedding(n + m + d + 1, 1, padding_idx=0)
        self.b = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.v.weight, std=0.03)
        nn.init.zeros_(self.w.weight)
        with torch.no_grad():
            self.v.weight[0].zero_()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        b = a.tocoo()
        seen = set((b.row.astype(np.int64) * m + b.col).tolist())
        pu = b.row[b.data >= 4]
        self.losses = []
        for _ in range(self.epochs):
            ni = rng.integers(m, size=len(pu))
            bad = np.fromiter((int(u) * m + int(i) in seen for u, i in zip(pu, ni)), dtype=bool)
            while bad.any():
                ni[bad] = rng.integers(m, size=bad.sum())
                bad = np.fromiter((int(u) * m + int(i) in seen for u, i in zip(pu, ni)), dtype=bool)
            u = torch.tensor(np.r_[b.row, pu], dtype=torch.long)
            i = torch.tensor(np.r_[b.col, ni], dtype=torch.long)
            t = torch.tensor(np.r_[b.data >= 4, np.zeros(len(pu))], dtype=torch.float32)
            order = rng.permutation(len(t))
            total = 0
            for start in range(0, len(t), 4096):
                j = order[start:start + 4096]
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self(u[j], i[j]), t[j])
                loss.backward()
                opt.step()
                total += loss.item() * len(j)
            self.losses.append(total / len(t))
        with torch.no_grad():
            v = self.v(self.f)
            self.iv = v.sum(dim=1)
            self.ib = self.w(self.f).sum(dim=(1, 2)) + 0.5 * (self.iv.square() - v.square().sum(dim=1)).sum(dim=1)
        return self

    def score(self, u):
        with torch.no_grad():
            u = torch.as_tensor(u, dtype=torch.long) + 1
            s = self.v(u) @ self.iv.T + self.w(u) + self.ib[None, :] + self.b
            return s.numpy()
