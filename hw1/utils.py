import ast
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import sparse


URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
FILES = ["reviews_Video_Games_5.json.gz", "meta_Video_Games.json.gz"]
HASHES = {
    FILES[0]: "035690169fa2bba30cd0ae0c3e7dbf7bc58efde65fc1742a9ce85986e283b09c",
    FILES[1]: "b7bf4c724632874b931056e48957775822c6f0c9092f5bbf2d449694bec36180",
}


def download():
    p = Path("data")
    p.mkdir(exist_ok=True)
    for s in FILES:
        f = p / s
        if not f.exists():
            urlretrieve(URL + s, f)
    hashes = {}
    for s in FILES:
        with (p / s).open("rb") as f:
            hashes[s] = hashlib.file_digest(f, "sha256").hexdigest()
        if hashes[s] != HASHES[s]:
            raise ValueError(f"Изменился исходный архив: {s}")
    return hashes


def load_data():
    hashes = download()
    rows = []
    with gzip.open(Path("data") / FILES[0], "rt") as f:
        for s in f:
            x = json.loads(s)
            rows.append([x.get(k) for k in ["reviewerID", "asin", "overall", "unixReviewTime"]])
    a = pd.DataFrame(rows, columns=["user", "item", "r", "t"])
    rows = []
    with gzip.open(Path("data") / FILES[1], "rt") as f:
        for s in f:
            x = ast.literal_eval(s)
            c = sorted({v for path in x.get("categories", []) for v in path if v != "Video Games"})
            rows.append([x.get("asin"), x.get("title"), c])
    b = pd.DataFrame(rows, columns=["item", "title", "cats"])
    return a, b, hashes


def prepare(a, b, t1, t2, n=3000):
    log = [["Исходные отзывы", len(a), a.user.nunique(), a.item.nunique()]]
    a = a.dropna().copy()
    a = a[a.r.between(1, 5) & (a.t > 0)]
    a = a.sort_values(["t", "user", "item"], kind="stable").drop_duplicates(["user", "item"], keep="first")
    b = b.dropna(subset=["item"]).drop_duplicates("item", keep="first")
    z = a[a.t < t1].groupby("item").size().rename("n").reset_index()
    z = z.sort_values(["n", "item"], ascending=[False, True]).head(n)
    a = a[a.item.isin(z.item)].copy()
    log.append([f"Отбор {n} товаров по train", len(a), a.user.nunique(), a.item.nunique()])
    z = a[a.t < t1].groupby("user").size()
    a = a[a.user.isin(z[z >= 3].index)].copy()
    us = sorted(a.user.unique())
    it = sorted(a.item.unique())
    a["u"] = pd.Categorical(a.user, categories=us).codes
    a["i"] = pd.Categorical(a.item, categories=it).codes
    b = b.set_index("item").reindex(it).reset_index()
    b["title"] = b.title.fillna(b.item)
    b["cats"] = b.cats.map(lambda x: x if isinstance(x, list) and x else ["Unknown"])
    vocab = sorted({s for x in b.cats for s in x})
    enc = {s: j for j, s in enumerate(vocab)}
    cats = [[enc[s] for s in x] for x in b.cats]
    log.append(["Пользователи с >=3 отзывами в train", len(a), len(us), len(it)])
    tr = a[a.t < t1].copy()
    va = a[(a.t >= t1) & (a.t < t2)].copy()
    te = a[a.t >= t2].copy()
    assert 100000 <= len(a) <= 2000000
    assert len(us) >= 1000 and len(it) >= 1000
    assert tr.t.max() < va.t.min() and va.t.max() < te.t.min()
    assert tr.u.nunique() == len(us) and tr.i.nunique() == len(it)
    log = pd.DataFrame(log, columns=["этап", "взаимодействия", "пользователи", "товары"])
    return a, b, cats, tr, va, te, log


def matrix(a, shape, positive=False):
    if positive:
        a = a[a.r >= 4]
    v = np.ones(len(a)) if positive else a.r.to_numpy()
    return sparse.csr_matrix((v, (a.u, a.i)), shape=shape, dtype=np.float64)


def truth(a, known, max_users=3000, seed=42):
    a = a[a.r >= 4]
    q = a.groupby("u").i.apply(set).to_dict()
    q = {u: v - set(known[u].indices) for u, v in q.items()}
    q = {u: v for u, v in q.items() if v}
    u = np.array(sorted(q), dtype=int)
    if len(u) > max_users:
        u = np.sort(np.random.default_rng(seed).choice(u, max_users, replace=False))
    return u, {int(x): q[int(x)] for x in u}


def recommend(model, u, known, k=10):
    s = np.asarray(model.score(u), dtype=np.float64)
    for j, x in enumerate(u):
        s[j, known[x].indices] = -np.inf
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def evaluate(model, u, q, known, k=10):
    rows = []
    for start in range(0, len(u), 128):
        v = u[start:start + 128]
        pred = recommend(model, v, known, k)
        for x, p in zip(v, pred):
            rel = q[int(x)]
            h = np.array([i in rel for i in p])
            dcg = np.sum(h / np.log2(np.arange(k) + 2))
            idcg = np.sum(1 / np.log2(np.arange(min(k, len(rel))) + 2))
            rows.append([int(x), h.sum() / len(rel), dcg / idcg])
    out = pd.DataFrame(rows, columns=["u", "Recall@10", "NDCG@10"])
    return out.iloc[:, 1:].mean().to_dict(), out


def interval(a, b, col="NDCG@10", seed=42):
    c = a.merge(b, on="u", suffixes=("_a", "_b"), validate="one_to_one")
    d = (c[col + "_b"] - c[col + "_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    z = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(2000)])
    return {"разность": d.mean(), "нижняя 95%": np.quantile(z, 0.025), "верхняя 95%": np.quantile(z, 0.975)}
