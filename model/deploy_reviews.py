"""A 재배포 — deploy 후보게임을 리뷰 enrich 모델로 재인코딩.

deploy/game_emb_appids.csv 순서를 유지한 채 reviews-content-model로 콘텐츠 임베딩 재생성
→ deploy/game_emb.npy 덮어쓰기(순서·차원 동일이라 app.py 무변경).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.train_reviews import load_docs_with_reviews  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL = ROOT / "models" / "reviews-content-model"
APPIDS = ROOT / "deploy" / "game_emb_appids.csv"
OUT = ROOT / "deploy" / "game_emb.npy"


def main():
    appids = pd.read_csv(APPIDS).iloc[:, 0].tolist()
    doc = load_docs_with_reviews()
    games = pd.read_parquet(ROOT / "deploy" / "games_lookup.parquet")
    name = dict(zip(games["appid"], games["name"]))
    texts, miss = [], 0
    for a in appids:
        d = doc.get(a)
        if not d:
            d = str(name.get(a, "")); miss += 1
        texts.append(d)
    print(f"deploy 게임 {len(appids):,}  (doc 없음 {miss} → 이름 폴백)")
    model = SentenceTransformer(str(MODEL))
    model.max_seq_length = 192
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    assert emb.shape[0] == len(appids), (emb.shape, len(appids))
    np.save(OUT, emb)
    print(f"저장: {OUT}  shape={emb.shape}")


if __name__ == "__main__":
    main()
