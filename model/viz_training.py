"""학습 과정 시각화용 데이터 생성 — 대조학습(item2vec) 임베딩의 에폭별 스냅샷.

TF Playground 스타일: 학습이 진행되며 게임 임베딩 점들이 군집을 형성하는 궤적을 캡처.
경량 설정(상위 ~1200 게임, dim 32)으로 빠르게 학습하며 에폭마다 임베딩 저장 →
최종 임베딩에 맞춘 PCA로 모든 에폭을 같은 2D 평면에 투영(정렬) → frames JSON.
산출: viz/training_frames.json  (데모/HTML 애니메이션용)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "dataset" / "games.parquet"
PAIRS = ROOT / "dataset" / "pairs.parquet"
OUT = ROOT / "viz"

N_ITEMS = 1200
DIM = 32
EPOCHS = 24
BATCH = 1024
TEMP = 0.07
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    g = pd.read_parquet(GAMES, columns=["appid", "name", "genres", "recommendations_total"])
    g = g.sort_values("recommendations_total", ascending=False).head(N_ITEMS).reset_index(drop=True)
    appids = g["appid"].tolist()
    idx = {a: i for i, a in enumerate(appids)}
    names = g["name"].tolist()

    def g0(s):
        try:
            xs = json.loads(s) if s else []
            return xs[0] if xs else "기타"
        except Exception:
            return "기타"
    genres = [g0(s) for s in g["genres"]]

    pairs = pd.read_parquet(PAIRS)
    pairs = pairs[pairs["appid_a"].isin(idx) & pairs["appid_b"].isin(idx)]
    a = pairs["appid_a"].map(idx).to_numpy()
    b = pairs["appid_b"].map(idx).to_numpy()
    P = torch.tensor(np.stack([a, b], 1), dtype=torch.long, device=DEVICE)
    print(f"items {len(appids)} · pairs {len(P):,} · device {DEVICE}")

    emb = nn.Embedding(len(appids), DIM).to(DEVICE)
    nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=2e-3)

    snaps, losses = [], []
    snaps.append(emb.weight.detach().cpu().numpy().copy())  # epoch 0 (랜덤)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(P.size(0), device=DEVICE)
        tot = 0.0
        for i in range(0, P.size(0), BATCH):
            ix = perm[i:i + BATCH]
            ea = F.normalize(emb(P[ix, 0]), dim=1)
            eb = F.normalize(emb(P[ix, 1]), dim=1)
            logits = ea @ eb.t() / TEMP
            lab = torch.arange(ea.size(0), device=DEVICE)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        losses.append(round(tot / (P.size(0) // BATCH + 1), 4))
        snaps.append(emb.weight.detach().cpu().numpy().copy())
    print("loss:", losses[0], "→", losses[-1])

    # 최종 임베딩에 PCA 맞추고 모든 에폭을 같은 평면에 투영(정렬)
    pca = PCA(n_components=2, random_state=0).fit(snaps[-1])
    frames = []
    for s in snaps:
        xy = pca.transform(s)
        frames.append([[round(float(x), 3), round(float(y), 3)] for x, y in xy])
    # 공통 스케일 정규화(애니메이션 축 고정용)
    allxy = np.array([p for f in frames for p in f])
    lo, hi = allxy.min(0), allxy.max(0)

    OUT.mkdir(exist_ok=True)
    (OUT / "training_frames.json").write_text(json.dumps({
        "names": names, "genres": genres, "frames": frames, "losses": losses,
        "bounds": {"xlo": float(lo[0]), "xhi": float(hi[0]), "ylo": float(lo[1]), "yhi": float(hi[1])},
        "epochs": EPOCHS, "dim": DIM, "n_items": len(appids),
    }, ensure_ascii=False), encoding="utf-8")
    print("저장: viz/training_frames.json | 프레임", len(frames), "· 게임", len(names))


if __name__ == "__main__":
    main()
