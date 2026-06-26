"""하이브리드 앙상블 평가 — 학습 협업 임베딩 + co-occurrence 결합.

두 신호의 오류 특성이 다름(임베딩=일반화·밀집, cooc=정확·고차원)을 이용해
점수를 정규화 후 가중 결합(α 스윕). 베이스라인(0.196) 초과 여부 확인.
동일 leave-one-out 프로토콜(seed0, 5천 유저, 후보 12k).
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
ALPHAS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]  # 0=cooc만, 1=임베딩만


def norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


def main():
    df = duckdb.query(
        f"SELECT steamid, appid FROM read_parquet('{INTER}') WHERE voted_up=1"
    ).df()
    pop = df.groupby("appid").size().sort_values(ascending=False)
    top = list(pop.head(TOP_ITEMS).index)
    iidx = {a: i for i, a in enumerate(top)}
    df = df[df["appid"].isin(set(top))]
    df["u"] = df["steamid"].map({u: i for i, u in enumerate(df["steamid"].unique())}).astype(np.int32)
    df["i"] = df["appid"].map(iidx).astype(np.int32)

    # cooc 행렬
    X = csr_matrix((np.ones(len(df), np.float32), (df["u"], df["i"])),
                   shape=(df["u"].max() + 1, len(top)))
    C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros()

    # 학습 임베딩 (appid 기준 정렬 정합)
    emb_raw = np.load(EMB)
    emb_ids = pd.read_csv(EMB_IDS)["appid"].tolist()
    id2row = {a: r for r, a in enumerate(emb_ids)}
    E = np.zeros((len(top), emb_raw.shape[1]), np.float32)
    for a, i in iidx.items():
        if a in id2row:
            E[i] = emb_raw[id2row[a]]

    # 평가셋
    by_user = df.groupby("u")["i"].apply(list)
    rng = random.Random(SEED)
    pool = [u for u, its in by_user.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool)
    eu = pool[:N_EVAL]
    holdout = {u: rng.choice(by_user[u]) for u in eu}
    profiles = {u: [i for i in by_user[u] if i != holdout[u]] for u in eu}

    maxk = max(KS)

    def run(alpha):
        agg, n = {}, 0
        for u in eu:
            prof = profiles[u]
            cooc_s = np.asarray(C[prof].sum(axis=0)).ravel()
            q = E[prof].mean(0); q /= (np.linalg.norm(q) + 1e-9)
            emb_s = E @ q
            s = (1 - alpha) * norm(cooc_s) + alpha * norm(emb_s)
            s[prof] = -np.inf
            topk = np.argpartition(-s, maxk)[:maxk]
            topk = topk[np.argsort(-s[topk])].tolist()
            tgt = holdout[u]
            pos = topk.index(tgt) if tgt in topk else None
            for k in KS:
                hit = pos is not None and pos < k
                agg[f"recall@{k}"] = agg.get(f"recall@{k}", 0) + (1.0 if hit else 0)
            agg["ndcg@10"] = agg.get("ndcg@10", 0) + ((1/math.log2(pos+2)) if (pos is not None and pos < 10) else 0)
            agg["mrr"] = agg.get("mrr", 0) + ((1/(pos+1)) if pos is not None else 0)
            n += 1
        return {k: round(v / n, 4) for k, v in agg.items()}

    print("alpha |  R@10   R@50   nDCG@10  MRR")
    for a in ALPHAS:
        r = run(a)
        print(f"{a:4.2f}  | {r['recall@10']:.4f} {r['recall@50']:.4f} {r['ndcg@10']:.4f}  {r['mrr']:.4f}")


if __name__ == "__main__":
    main()
