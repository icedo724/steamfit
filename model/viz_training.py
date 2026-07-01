"""학습 과정 시각화용 데이터 생성 — 대조학습(item2vec) 임베딩의 에폭별 스냅샷.

TF Playground 스타일: 학습이 진행되며 게임 임베딩 점들이 군집을 형성하는 궤적을 캡처.
경량 설정(상위 게임)으로 학습하며 에폭마다 임베딩 저장.
투영: 최종 임베딩에 UMAP을 맞추고(진짜 군집 분리) 각 스냅샷을 같은 공간에 transform.
프레임: 손실 수렴이 빨라 뒤쪽이 평평 → '학습 진행도' 기준으로 선별(흩어짐→뭉침이 크게 보이게).
색: 협업 이웃 KMeans '동네'(장르가 Action 편중이라 군집이 더 잘 보임).
산출: viz/training_frames.json (데모/HTML 애니메이션용)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "dataset" / "games.parquet"
PAIRS = ROOT / "dataset" / "pairs.parquet"
OUT = ROOT / "viz"

N_ITEMS = 900
DIM = 32
EPOCHS = 24
BATCH = 1024
TEMP = 0.07
N_FRAMES = 12
N_CLUSTERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    g = pd.read_parquet(GAMES, columns=["appid", "name", "genres", "recommendations_total"])
    g = g.sort_values("recommendations_total", ascending=False).head(N_ITEMS).reset_index(drop=True)
    appids = g["appid"].tolist()
    idx = {a: i for i, a in enumerate(appids)}
    names = g["name"].tolist()

    pairs = pd.read_parquet(PAIRS)
    pairs = pairs[pairs["appid_a"].isin(idx) & pairs["appid_b"].isin(idx)]
    a = pairs["appid_a"].map(idx).to_numpy()
    b = pairs["appid_b"].map(idx).to_numpy()
    P = torch.tensor(np.stack([a, b], 1), dtype=torch.long, device=DEVICE)
    print(f"items {len(appids)} · pairs {len(P):,} · device {DEVICE}")

    emb = nn.Embedding(len(appids), DIM).to(DEVICE)
    nn.init.normal_(emb.weight, std=0.35)          # 확산된 초기(흩어짐 강조)
    opt = torch.optim.Adam(emb.parameters(), lr=2e-3)

    snaps, losses = [], [None]
    snaps.append(emb.weight.detach().cpu().numpy().copy())            # epoch 0 (랜덤)
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
    print("loss:", losses[1], "→", losses[-1])

    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    # 최종(정규화)에 UMAP 맞추고 모든 스냅샷을 같은 공간으로 transform → 실제 궤적
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.28, random_state=42, metric="cosine")
    final2d = reducer.fit_transform(norm(snaps[-1]))
    clusters = KMeans(N_CLUSTERS, n_init=10, random_state=0).fit_predict(norm(snaps[-1]))

    # 학습 진행도(손실 하락 비율) 기준으로 프레임 선별 — 평평한 꼬리 제거, 앞쪽 촘촘
    L = np.array([losses[1]] + losses[1:], float)          # ep0엔 ep1 손실 대입(진행도 0 기준용)
    prog = (L[0] - L) / (L[0] - L[-1] + 1e-9)
    prog[0] = 0.0
    targets = np.linspace(0, 1, N_FRAMES)
    sel = sorted({int(np.argmin(np.abs(prog - t))) for t in targets})
    if sel[0] != 0:
        sel = [0] + sel

    frames = [reducer.transform(norm(snaps[i])) for i in sel]
    frames = np.array(frames)                              # [F, N, 2]
    # 공통 스케일: 전체 min/max로 [-1,1] 정규화
    lo, hi = frames.reshape(-1, 2).min(0), frames.reshape(-1, 2).max(0)
    frames = 2 * (frames - lo) / (hi - lo + 1e-9) - 1
    frames_out = [[[round(float(x), 3), round(float(y), 3)] for x, y in f] for f in frames]
    sel_loss = [losses[i] if i > 0 else None for i in sel]

    # 프레임 간 실제 이동량(검증)
    mov = [round(float(np.mean(np.linalg.norm(frames[i + 1] - frames[i], axis=1))), 3)
           for i in range(len(frames) - 1)]
    print("선별 스냅샷(epoch):", sel)
    print("프레임 간 이동량:", mov)
    print(f"프레임0 spread={np.std(frames[0]):.3f}  최종 spread={np.std(frames[-1]):.3f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "training_frames.json").write_text(json.dumps({
        "names": names, "clusters": clusters.tolist(),
        "frames": frames_out, "losses": sel_loss, "sel_epochs": sel,
        "bounds": {"xlo": -1.0, "xhi": 1.0, "ylo": -1.0, "yhi": 1.0},
        "epochs": EPOCHS, "dim": DIM, "n_items": len(appids), "n_clusters": N_CLUSTERS,
    }, ensure_ascii=False), encoding="utf-8")
    print("저장: viz/training_frames.json | 프레임", len(frames_out), "· 게임", len(names))


if __name__ == "__main__":
    main()
