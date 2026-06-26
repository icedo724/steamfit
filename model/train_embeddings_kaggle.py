"""티어 2·3 — 콘텐츠 임베딩 (사전학습 → 대조학습 파인튜닝). [Kaggle GPU에서 실행]

이 프로젝트의 주인공. 같은 leave-one-out 프로토콜로 평가해
로컬 베이스라인(co-occurrence Recall@10≈0.196)과 직접 비교한다.

흐름:
  1) 게임 doc을 사전학습 임베딩으로 인코딩 → 평가 (티어2: 파인튜닝 전 콘텐츠)
  2) 양성쌍(같이 즐겨진 게임)으로 대조학습(MultipleNegativesRankingLoss) 파인튜닝
  3) 다시 인코딩 → 평가 (티어3: 파인튜닝 후) → 향상폭 확인
  4) 모델 + 게임 임베딩 저장 (데모용)

Kaggle 사용:
  - dataset/games.parquet, pairs.parquet, interactions.parquet 를 Kaggle Dataset으로 업로드
  - GPU 노트북에서 이 스크립트 실행 (pip install sentence-transformers)
  - DATA_DIR 를 업로드 경로(/kaggle/input/<name>)로 지정
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ── 경로 (Kaggle 입력 경로로 교체) ────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/steam-recsys")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 영어 doc에 충분·빠름

# ── 평가 프로토콜 (로컬 베이스라인과 동일하게 고정) ───────────────
TOP_ITEMS = 12_000
N_EVAL = 5_000
MIN_ITEMS = 5
KS = (10, 50)
SEED = 0

EPOCHS = 1
BATCH = 128


def load_frames():
    games = pd.read_parquet(f"{DATA_DIR}/games.parquet")
    pairs = pd.read_parquet(f"{DATA_DIR}/pairs.parquet")
    inter = pd.read_parquet(f"{DATA_DIR}/interactions.parquet",
                            columns=["steamid", "appid", "voted_up"])
    inter = inter[inter["voted_up"] == 1]
    return games, pairs, inter


def build_eval(inter):
    """leave-one-out 평가셋 (eval_baseline.py와 동일 규칙·시드)."""
    pop = inter.groupby("appid").size().sort_values(ascending=False)
    top = list(pop.head(TOP_ITEMS).index)
    top_set = set(top)
    inter = inter[inter["appid"].isin(top_set)]

    by_user = inter.groupby("steamid")["appid"].apply(list)
    rng = random.Random(SEED)
    pool = [u for u, its in by_user.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool)
    eval_users = pool[:N_EVAL]
    holdout = {u: rng.choice(by_user[u]) for u in eval_users}
    profiles = {u: [a for a in by_user[u] if a != holdout[u]] for u in eval_users}
    return top, holdout, profiles, eval_users


def encode_games(model, games, candidate_appids):
    sub = games[games["appid"].isin(set(candidate_appids))][["appid", "doc"]]
    emb = model.encode(sub["doc"].tolist(), batch_size=256, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    appid2row = {a: i for i, a in enumerate(sub["appid"].tolist())}
    return emb.astype(np.float32), appid2row


def evaluate(emb, appid2row, holdout, profiles, eval_users):
    maxk = max(KS)
    agg = {}
    n = 0
    for u in eval_users:
        prof = [appid2row[a] for a in profiles[u] if a in appid2row]
        tgt = holdout[u]
        if not prof or tgt not in appid2row:
            continue
        q = emb[prof].mean(axis=0)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = emb @ q
        scores[prof] = -np.inf
        topk = np.argpartition(-scores, maxk)[:maxk]
        topk = topk[np.argsort(-scores[topk])].tolist()
        tr = appid2row[tgt]
        pos = topk.index(tr) if tr in topk else None
        for k in KS:
            hit = pos is not None and pos < k
            agg[f"recall@{k}"] = agg.get(f"recall@{k}", 0) + (1.0 if hit else 0)
            agg[f"ndcg@{k}"] = agg.get(f"ndcg@{k}", 0) + ((1/math.log2(pos+2)) if hit else 0)
        agg["mrr"] = agg.get("mrr", 0) + ((1/(pos+1)) if pos is not None else 0)
        n += 1
    return {k: round(v / n, 4) for k, v in agg.items()}


def main():
    games, pairs, inter = load_frames()
    top, holdout, profiles, eval_users = build_eval(inter)
    print(f"평가 유저 {len(eval_users):,} · 후보 게임 {len(top):,}")

    model = SentenceTransformer(BASE_MODEL)

    # 티어2: 파인튜닝 전 콘텐츠 임베딩
    emb0, a2r = encode_games(model, games, top)
    print("티어2 (사전학습 콘텐츠):", evaluate(emb0, a2r, holdout, profiles, eval_users))

    # 대조학습 파인튜닝 (양성쌍 → in-batch negatives)
    doc = dict(zip(games["appid"], games["doc"]))
    examples = [
        InputExample(texts=[doc[a], doc[b]])
        for a, b in zip(pairs["appid_a"], pairs["appid_b"])
        if a in doc and b in doc
    ]
    print(f"학습 쌍 {len(examples):,}")
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=int(0.1 * len(loader)), show_progress_bar=True)

    # 티어3: 파인튜닝 후
    emb1, a2r = encode_games(model, games, top)
    print("티어3 (대조학습 파인튜닝):", evaluate(emb1, a2r, holdout, profiles, eval_users))

    # 저장 (데모용)
    model.save(f"{OUT_DIR}/steam-embed-model")
    np.save(f"{OUT_DIR}/game_emb.npy", emb1)
    pd.Series(list(a2r.keys())).to_csv(f"{OUT_DIR}/game_emb_appids.csv", index=False)
    print("저장 완료:", OUT_DIR)


if __name__ == "__main__":
    main()
