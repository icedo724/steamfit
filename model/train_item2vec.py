"""티어 3b — 협업 임베딩 직접 학습 (item2vec식 대조학습). [GPU]

게임 ID 임베딩 테이블을 공동플레이 양성쌍으로 대조학습(in-batch negatives)한다.
콘텐츠(텍스트)가 아니라 행동 신호를 직접 학습하므로, 같은 행동 기반 평가에서
co-occurrence 베이스라인(Recall@10=0.196)을 넘는 것이 목표.

동일 leave-one-out 프로토콜(평가유저 5천·후보 12k·seed0)로 직접 비교.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
INTER = (ROOT / "dataset" / "interactions.parquet").as_posix()
PAIRS = (ROOT / "dataset" / "pairs.parquet").as_posix()
OUT = ROOT / "models"

TOP_ITEMS = 12_000
N_EVAL = 5_000
MIN_ITEMS = 5
KS = (10, 50)
SEED = 0

DIM = 256
EPOCHS = 30
BATCH = 2048
LR = 1e-3
TEMP = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    con = duckdb.connect()
    inter = con.execute(
        f"SELECT steamid, appid FROM read_parquet('{INTER}') WHERE voted_up=1"
    ).df()
    pop = inter.groupby("appid").size().sort_values(ascending=False)
    top = list(pop.head(TOP_ITEMS).index)
    iidx = {a: i for i, a in enumerate(top)}
    top_set = set(top)
    inter = inter[inter["appid"].isin(top_set)]

    pairs = con.execute(f"SELECT appid_a, appid_b FROM read_parquet('{PAIRS}')").df()
    pairs = pairs[pairs["appid_a"].isin(top_set) & pairs["appid_b"].isin(top_set)]
    a = pairs["appid_a"].map(iidx).to_numpy()
    b = pairs["appid_b"].map(iidx).to_numpy()
    con.close()
    return inter, top, iidx, np.stack([a, b], 1).astype(np.int64)


def make_eval(inter, iidx):
    by_user = inter.groupby("steamid")["appid"].apply(list)
    rng = random.Random(SEED)
    pool = [u for u, its in by_user.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool)
    eu = pool[:N_EVAL]
    holdout = {u: iidx[rng.choice(by_user[u])] for u in eu}
    profiles = {u: [iidx[a] for a in by_user[u] if iidx[a] != holdout[u]] for u in eu}
    return eu, holdout, profiles


def evaluate(emb, eu, holdout, profiles):
    maxk = max(KS)
    agg, n = {}, 0
    for u in eu:
        prof = profiles[u]
        if not prof:
            continue
        q = emb[prof].mean(0)
        q /= (np.linalg.norm(q) + 1e-9)
        s = emb @ q
        s[prof] = -np.inf
        topk = np.argpartition(-s, maxk)[:maxk]
        topk = topk[np.argsort(-s[topk])].tolist()
        tgt = holdout[u]
        pos = topk.index(tgt) if tgt in topk else None
        for k in KS:
            hit = pos is not None and pos < k
            agg[f"recall@{k}"] = agg.get(f"recall@{k}", 0) + (1.0 if hit else 0)
            agg[f"ndcg@{k}"] = agg.get(f"ndcg@{k}", 0) + ((1 / math.log2(pos + 2)) if hit else 0)
        agg["mrr"] = agg.get("mrr", 0) + ((1 / (pos + 1)) if pos is not None else 0)
        n += 1
    return {k: round(v / n, 4) for k, v in agg.items()}


def main():
    inter, top, iidx, pairs = load()
    n_items = len(top)
    print(f"아이템 {n_items:,} · 학습쌍 {len(pairs):,} · device={DEVICE}")
    eu, holdout, profiles = make_eval(inter, iidx)

    emb = nn.Embedding(n_items, DIM).to(DEVICE)
    nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=LR)
    P = torch.from_numpy(pairs).to(DEVICE)

    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(P.size(0), device=DEVICE)
        tot = 0.0
        for i in range(0, P.size(0), BATCH):
            idx = perm[i:i + BATCH]
            a, b = P[idx, 0], P[idx, 1]
            ea = F.normalize(emb(a), dim=1)
            eb = F.normalize(emb(b), dim=1)
            logits = ea @ eb.t() / TEMP
            labels = torch.arange(a.size(0), device=DEVICE)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        vecs = F.normalize(emb.weight.detach(), dim=1).cpu().numpy()
        print(f"  epoch {ep} loss={tot/(P.size(0)//BATCH+1):.4f} | {evaluate(vecs, eu, holdout, profiles)}")

    vecs = F.normalize(emb.weight.detach(), dim=1).cpu().numpy().astype(np.float32)
    print("티어3b (협업 임베딩):", evaluate(vecs, eu, holdout, profiles))
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "item2vec_emb.npy", vecs)
    import pandas as pd
    pd.Series(top, name="appid").to_csv(OUT / "item2vec_appids.csv", index=False)
    print("저장: models/item2vec_emb.npy")


if __name__ == "__main__":
    main()
