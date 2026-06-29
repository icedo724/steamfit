"""취향 경로 개선 측정 + cooc 이웃 추출.

데모의 현재 취향(item2vec 단독)을 cooc·하이브리드(가중/RRF) 융합과 동일 평가로 비교.
+ 데모에 넣을 item-item cooc top-K 이웃 저장(deploy/cooc_topk.npz).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
INTER = (ROOT / "dataset" / "interactions.parquet").as_posix()
EMB = ROOT / "models" / "item2vec_emb.npy"
EMB_IDS = ROOT / "models" / "item2vec_appids.csv"
TOP_ITEMS, N_EVAL, MIN_ITEMS, SEED = 12_000, 5_000, 5, 0
KS = (10, 50)
TOPK_SAVE = 200


def main():
    df = duckdb.query(f"SELECT steamid, appid FROM read_parquet('{INTER}') WHERE voted_up=1").df()
    pop = df.groupby("appid").size().sort_values(ascending=False)
    top = list(pop.head(TOP_ITEMS).index)
    iidx = {a: i for i, a in enumerate(top)}
    df = df[df["appid"].isin(set(top))]
    uids = {u: i for i, u in enumerate(df["steamid"].unique())}
    df["u"] = df["steamid"].map(uids).astype(np.int32)
    df["i"] = df["appid"].map(iidx).astype(np.int32)

    X = csr_matrix((np.ones(len(df), np.float32), (df["u"], df["i"])), shape=(len(uids), len(top)))
    C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros()

    E = np.zeros((len(top), 0))
    emb_raw = np.load(EMB)
    ids = pd.read_csv(EMB_IDS)["appid"].tolist()
    id2r = {a: r for r, a in enumerate(ids)}
    E = np.zeros((len(top), emb_raw.shape[1]), np.float32)
    for a, i in iidx.items():
        if a in id2r:
            E[i] = emb_raw[id2r[a]]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    by = df.groupby("u")["i"].apply(list)
    rng = random.Random(SEED)
    pool = [u for u, its in by.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool); eu = pool[:N_EVAL]
    holdout = {u: rng.choice(by[u]) for u in eu}
    prof = {u: [i for i in by[u] if i != holdout[u]] for u in eu}
    mk = max(KS)

    def emb_s(p): q = En[p].mean(0); return En @ (q / (np.linalg.norm(q) + 1e-9))
    def cooc_s(p): return np.asarray(C[p].sum(0)).ravel()

    def run(scorer):
        agg = {}; n = 0
        for u in eu:
            p = prof[u]
            s = scorer(p).astype(float); s[p] = -np.inf
            tk = np.argpartition(-s, mk)[:mk]; tk = tk[np.argsort(-s[tk])].tolist()
            pos = tk.index(holdout[u]) if holdout[u] in tk else None
            for k in KS:
                hit = pos is not None and pos < k
                agg[f"R@{k}"] = agg.get(f"R@{k}", 0) + (1 if hit else 0)
            agg["nDCG@10"] = agg.get("nDCG@10", 0) + ((1/math.log2(pos+2)) if (pos is not None and pos < 10) else 0)
            agg["MRR"] = agg.get("MRR", 0) + ((1/(pos+1)) if pos is not None else 0)
            n += 1
        return {k: round(v/n, 4) for k, v in agg.items()}

    def nz(v): return (v - v.min()) / (v.max() - v.min() + 1e-9)
    def weighted(a):
        return lambda p: (1-a)*nz(cooc_s(p)) + a*nz(emb_s(p))
    def rrf(p, kk=60):
        re = rankdata(-emb_s(p)); rc = rankdata(-cooc_s(p))
        return 1/(kk+re) + 1/(kk+rc)

    print("=== 취향 경로 비교 (leave-one-out) ===")
    print("emb 단독(현 데모):", run(emb_s))
    print("cooc 단독       :", run(cooc_s))
    print("가중 α=0.8      :", run(weighted(0.8)))
    print("가중 α=0.65     :", run(weighted(0.65)))
    print("RRF             :", run(rrf))

    # cooc top-K 저장 (데모 하이브리드 취향용)
    Cc = C.tocsr()
    nb = np.zeros((len(top), TOPK_SAVE), np.int32)
    wt = np.zeros((len(top), TOPK_SAVE), np.float32)
    for i in range(len(top)):
        row = Cc.getrow(i)
        if row.nnz:
            idxs, vals = row.indices, row.data
            o = np.argsort(-vals)[:TOPK_SAVE]
            nb[i, :len(o)] = idxs[o]; wt[i, :len(o)] = vals[o]
    np.savez((ROOT / "deploy" / "cooc_topk.npz"), appids=np.array(top, np.int64), nb=nb, wt=wt)
    print("저장: deploy/cooc_topk.npz (top%d 이웃)" % TOPK_SAVE)


if __name__ == "__main__":
    main()
