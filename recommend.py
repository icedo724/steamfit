"""하이브리드 추천기 — 취향(협업 임베딩) + 의도(텍스트 콘텐츠 임베딩).

  taste  : 즐긴 게임들의 협업 임베딩(item2vec) 평균 → 후보와 코사인 (행동 신호)
  intent : 원하는 특징 텍스트 → 콘텐츠 임베딩 → 후보와 코사인 (Steam이 못 하는 의도 반영)
  final  = w_taste·norm(taste) + w_intent·norm(intent), 이미 한 게임 제외 후 정렬

CLI 예:
  python recommend.py --liked 400 620 --intent "relaxing open world crafting" --topn 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
M = ROOT / "models"


class Recommender:
    def __init__(self):
        games = pd.read_parquet(ROOT / "dataset" / "games.parquet",
                                columns=["appid", "name", "genres"])
        self.name = dict(zip(games["appid"], games["name"]))

        # 협업 임베딩 (취향)
        self.collab = np.load(M / "item2vec_emb.npy")
        cids = pd.read_csv(M / "item2vec_appids.csv")["appid"].tolist()
        self.c_row = {a: i for i, a in enumerate(cids)}

        # 콘텐츠 임베딩 (의도) + 텍스트 인코더
        self.content = np.load(M / "game_emb.npy")
        tids = pd.read_csv(M / "game_emb_appids.csv").iloc[:, 0].tolist()
        self.t_row = {a: i for i, a in enumerate(tids)}

        # 두 임베딩 모두 있는 게임만 후보로
        self.cand = [a for a in cids if a in self.t_row]
        self.collab_c = np.stack([self.collab[self.c_row[a]] for a in self.cand])
        self.content_c = np.stack([self.content[self.t_row[a]] for a in self.cand])
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(str(M / "steam-embed-model"))
        return self._encoder

    @staticmethod
    def _norm(v):
        return (v - v.min()) / (v.max() - v.min() + 1e-9)

    def recommend(self, liked, intent="", w_taste=0.6, w_intent=0.4, topn=10):
        n = len(self.cand)
        score = np.zeros(n, np.float32)

        liked_rows = [self.cand.index(a) for a in liked if a in self.c_row and a in self.t_row]
        if liked_rows:
            q = self.collab_c[liked_rows].mean(0)
            q /= np.linalg.norm(q) + 1e-9
            score += w_taste * self._norm(self.collab_c @ q)

        if intent.strip():
            qi = self.encoder.encode(intent, normalize_embeddings=True)
            score += w_intent * self._norm(self.content_c @ qi)

        for r in liked_rows:
            score[r] = -np.inf
        top = np.argsort(-score)[:topn]
        return [(self.cand[i], self.name.get(self.cand[i], "?"), float(score[i])) for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liked", type=int, nargs="*", default=[])
    ap.add_argument("--intent", default="")
    ap.add_argument("--w-taste", type=float, default=0.6)
    ap.add_argument("--w-intent", type=float, default=0.4)
    ap.add_argument("--topn", type=int, default=10)
    a = ap.parse_args()

    rec = Recommender()
    print(f"즐긴 게임: {[rec.name.get(x, x) for x in a.liked]}")
    print(f"의도: '{a.intent}'  (w_taste={a.w_taste}, w_intent={a.w_intent})\n")
    for i, (appid, name, s) in enumerate(rec.recommend(
            a.liked, a.intent, a.w_taste, a.w_intent, a.topn), 1):
        print(f"  {i:2d}. {name}  (score={s:.3f})")


if __name__ == "__main__":
    main()
