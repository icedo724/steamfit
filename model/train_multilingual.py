"""티어 — 다국어 콘텐츠 임베딩 (태그 enriched + EN/KO, 대조학습). [GPU]

목표: 한국어 의도("협동 호러")도 영어 태그("Co-op","Horror")와 교차언어 매칭.
- 베이스: paraphrase-multilingual-MiniLM-L12-v2 (다국어)
- 문서: 이름 + Steam태그 + 장르 + EN/KO 설명  (MotherDuck에서 로드)
- 공동플레이 양성쌍(pairs.parquet)으로 대조학습
동일 leave-one-out(seed0, 후보 12k)로 기존 콘텐츠 티어와 비교 + 한국어 의도 sanity check.
저장: models/ml-content-model, models/game_emb_ml.npy (+appids) → 데모 갱신용.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.db import connect

INTER = (ROOT / "dataset" / "interactions.parquet").as_posix()
PAIRS = (ROOT / "dataset" / "pairs.parquet").as_posix()
OUT = ROOT / "models"
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_ITEMS, N_EVAL, MIN_ITEMS, SEED = 12_000, 5_000, 5, 0
KS = (10, 50)
EPOCHS, BATCH = 1, 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def jl(s):
    try:
        return json.loads(s) if s else []
    except Exception:
        return []


def load_docs():
    """MotherDuck에서 게임 메타 → 다국어 doc (이름+태그+장르+EN/KO설명)."""
    con, where = connect()
    print("docs from:", where)
    df = con.execute("""
        SELECT appid, name, name_ko, tags, genres_en, short_desc_en, short_desc_ko
        FROM games WHERE type='game' AND COALESCE(recommendations_total,0) > 0
    """).df()
    con.close()
    doc = {}
    for r in df.itertuples():
        parts = [str(r.name or "")]
        parts += jl(r.tags)          # Steam 태그 (가장 풍부한 신호)
        parts += jl(r.genres_en)
        if r.short_desc_en:
            parts.append(str(r.short_desc_en))
        if r.short_desc_ko:
            parts.append(str(r.short_desc_ko))
        doc[r.appid] = " | ".join(p for p in parts if p).strip()
    return doc


def build_eval():
    df = pd.read_parquet(INTER, columns=["steamid", "appid", "voted_up"])
    df = df[df["voted_up"] == 1]
    pop = df.groupby("appid").size().sort_values(ascending=False)
    top = list(pop.head(TOP_ITEMS).index)
    df = df[df["appid"].isin(set(top))]
    by = df.groupby("steamid")["appid"].apply(list)
    rng = random.Random(SEED)
    pool = [u for u, its in by.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool)
    eu = pool[:N_EVAL]
    holdout = {u: rng.choice(by[u]) for u in eu}
    prof = {u: [a for a in by[u] if a != holdout[u]] for u in eu}
    return top, holdout, prof, eu


def encode(model, appids, doc):
    sub = [(a, doc[a]) for a in appids if a in doc and doc[a]]
    emb = model.encode([d for _, d in sub], batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float32), {a: i for i, (a, _) in enumerate(sub)}


def evaluate(emb, a2r, holdout, prof, eu):
    mk = max(KS); agg = {}; n = 0
    for u in eu:
        rows = [a2r[a] for a in prof[u] if a in a2r]
        tgt = holdout[u]
        if not rows or tgt not in a2r:
            continue
        q = emb[rows].mean(0); q /= np.linalg.norm(q) + 1e-9
        s = emb @ q; s[rows] = -np.inf
        topk = np.argpartition(-s, mk)[:mk]; topk = topk[np.argsort(-s[topk])].tolist()
        tr = a2r[tgt]; pos = topk.index(tr) if tr in topk else None
        for k in KS:
            hit = pos is not None and pos < k
            agg[f"recall@{k}"] = agg.get(f"recall@{k}", 0) + (1.0 if hit else 0)
        agg["ndcg@10"] = agg.get("ndcg@10", 0) + ((1/math.log2(pos+2)) if (pos is not None and pos < 10) else 0)
        n += 1
    return {k: round(v/n, 4) for k, v in agg.items()}


def ko_sanity(model, emb, a2r, doc, name, queries):
    r2a = {v: k for k, v in a2r.items()}
    for q in queries:
        qe = model.encode(q, normalize_embeddings=True)
        s = emb @ qe; top = np.argsort(-s)[:5]
        print(f"  '{q}' →", [name.get(r2a[i], r2a[i]) for i in top])


def main():
    doc = load_docs()
    print(f"docs: {len(doc):,}")
    top, holdout, prof, eu = build_eval()
    pairs = pd.read_parquet(PAIRS)
    if len(pairs) > 1_500_000:        # 다국어 모델은 무거우니 학습쌍 샘플링(시간 제한)
        pairs = pairs.sample(1_500_000, random_state=0)
    names = {a: d.split(" | ")[0] for a, d in doc.items()}

    model = SentenceTransformer(BASE, device=DEVICE)

    emb0, a2r = encode(model, top, doc)
    print("다국어 사전학습:", evaluate(emb0, a2r, holdout, prof, eu))

    ex = [InputExample(texts=[doc[a], doc[b]])
          for a, b in zip(pairs["appid_a"], pairs["appid_b"]) if a in doc and b in doc]
    print(f"학습쌍: {len(ex):,}")
    loader = DataLoader(ex, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=int(0.1*len(loader)), show_progress_bar=True)

    emb1, a2r = encode(model, top, doc)
    print("다국어 대조학습:", evaluate(emb1, a2r, holdout, prof, eu))

    print("한국어 의도 sanity:")
    ko_sanity(model, emb1, a2r, doc, names,
              ["협동 공포 게임", "여유로운 농장 경영", "어려운 소울라이크"])
    print("영어 의도 sanity:")
    ko_sanity(model, emb1, a2r, doc, names,
              ["co-op horror", "relaxing farming sim", "hard soulslike"])

    OUT.mkdir(exist_ok=True)
    model.save(str(OUT / "ml-content-model"))
    np.save(OUT / "game_emb_ml.npy", emb1)
    pd.Series(list(a2r.keys()), name="appid").to_csv(OUT / "game_emb_ml_appids.csv", index=False)
    print("저장 완료: models/ml-content-model, game_emb_ml.npy")


if __name__ == "__main__":
    main()
